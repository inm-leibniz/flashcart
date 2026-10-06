from typing import List, Optional, Union

import ase
import ase.io
import numpy as np
import torch

from flashcart.data.neighbors import NeighborList
from flashcart.data.utils import numbers_to_types
from flashcart.utils.geometry import uses_periodic_shifts
from flashcart.utils.torch_geometric.data import Data


class AtomicConfig:
    """Store one atomic configuration and its optional reference properties.

    The configuration stores NumPy arrays and attaches a neighbor list when
    ``compute_neighbors`` is called. Neighbor-list reuse depends on the stored cutoff
    and skin. Changes to the geometry do not invalidate the list automatically.

    Args:
        atomic_numbers (np.ndarray): Per-atom atomic numbers.
        positions (np.ndarray): Cartesian positions, shape (n, 3).
        cell (np.ndarray, optional): 3x3 cell matrix. Default: zeros.
        pbc (list[bool] | bool, optional): Periodic flags. Default: all False.
        energy (float, optional): Reference total energy. Default: None.
        forces (np.ndarray, optional): Reference forces, with shape ``(n_atoms, 3)``.
            Default: None.
        stress (np.ndarray, optional): Reference stress, with shape ``(3, 3)`` or
            ``(6,)`` in the order ``xx, yy, zz, yz, xz, xy``. Matrix inputs are
            symmetrized and converted to this order. Default: None.
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[Union[List[bool], bool]] = None,
        energy: Optional[float] = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[np.ndarray] = None,
    ):
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.cell = np.zeros((3, 3), dtype=np.float64) if cell is None else cell
        self.pbc = np.zeros(3, dtype=bool) if pbc is None else pbc

        if energy is not None:
            energy = float(energy)
        self.energy = energy

        if forces is not None:
            assert isinstance(forces, np.ndarray) and forces.shape == positions.shape, (
                "forces must be a numpy array with the same shape as positions. "
                f"Provided: {type(forces)} with shape {forces.shape}."
            )
        self.forces = forces

        if stress is not None:
            assert isinstance(stress, np.ndarray) and stress.shape in ((3, 3), (6,)), (
                "stress must be a numpy array with shape (3, 3) or (6,). "
                f"Provided: {type(stress)} with shape {stress.shape}."
            )
            if stress.shape == (3, 3):
                from flashcart.utils.geometry import to_voigt6

                stress = to_voigt6(stress)
        self.stress = stress

        self.neighbors: Optional[NeighborList] = None

    def compute_neighbors(self, r_max: float, skin: float = 0.0, backend: str = "matscipy") -> None:
        """Build or reuse the neighbor list at ``r_max + skin``.

        An existing list is reused when its cutoff and skin match the requested values.
        This check does not compare positions, cell, periodicity, or backend. Clear
        ``neighbors`` before calling this method if these have changed and a new list is
        required.

        Args:
            r_max (float): Cutoff radius.
            skin (float, optional): Extra radius for reuse. Default: 0.0.
            backend (str, optional): Neighbor backend. Default: "matscipy".
        """
        if (
            self.neighbors is not None
            and self.neighbors.is_built
            and self.neighbors.r_max == r_max
            and self.neighbors.skin == skin
        ):
            return
        self.neighbors = NeighborList(r_max=r_max, skin=skin, backend=backend).build(
            self.positions, self.cell, self.pbc
        )

    @classmethod
    def from_ase(
        cls,
        atoms: ase.Atoms,
        energy_key: str = "REF_energy",
        forces_key: str = "REF_forces",
        stress_key: str = "REF_stress",
    ) -> "AtomicConfig":
        """Read an ASE configuration and its stored reference properties.

        Reference energies and stresses are read from ``atoms.info`` and forces from
        ``atoms.arrays``. Missing properties are stored as None.

        Args:
            atoms (ase.Atoms): Atomic configuration to convert.
            energy_key (str, optional): Key for the reference energy in ``atoms.info``.
                Default: "REF_energy".
            forces_key (str, optional): Key for the reference forces in
                ``atoms.arrays``. Default: "REF_forces".
            stress_key (str, optional): Key for the reference stress in ``atoms.info``.
                Default: "REF_stress".

        Returns:
            AtomicConfig: Configuration containing the geometry and available reference
                properties.
        """
        return cls(
            atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int64),
            positions=np.asarray(atoms.get_positions(), dtype=np.float64),
            cell=np.asarray(atoms.get_cell(), dtype=np.float64),
            pbc=np.asarray(atoms.get_pbc(), dtype=bool),
            energy=float(atoms.info[energy_key]) if energy_key in atoms.info else None,
            forces=atoms.arrays.get(forces_key),
            stress=atoms.info.get(stress_key),
        )

    def to_ase(
        self,
        energy_key: str = "REF_energy",
        forces_key: str = "REF_forces",
        stress_key: str = "REF_stress",
    ) -> ase.Atoms:
        """Convert the configuration to ASE and store its reference properties.

        Only properties present in the configuration are written.

        Args:
            energy_key (str, optional): Key for the reference energy in ``atoms.info``.
                Default: "REF_energy".
            forces_key (str, optional): Key for the reference forces in
                ``atoms.arrays``. Default: "REF_forces".
            stress_key (str, optional): Key for the reference stress in ``atoms.info``.
                Default: "REF_stress".

        Returns:
            ase.Atoms: Atomic configuration with the available reference properties
                stored under the specified keys.
        """
        atoms = ase.Atoms(
            numbers=self.atomic_numbers.astype(np.int64),
            positions=self.positions.astype(np.float64),
            cell=self.cell.astype(np.float64),
            pbc=self.pbc.astype(bool),
        )
        if self.energy is not None:
            atoms.info[energy_key] = float(self.energy)
        if self.forces is not None:
            atoms.arrays[forces_key] = self.forces.astype(np.float64)
        if self.stress is not None:
            atoms.info[stress_key] = self.stress.astype(np.float64)
        return atoms


class AtomicData(Data):
    """Represent one atomic configuration as a graph of PyTorch tensors.

    Floating-point fields use ``torch.get_default_dtype()``. Connectivity, atomic
    numbers, atom types, and periodic shifts use integer tensors. The shifts specify
    translations in units of the cell vectors. Cartesian translations are formed when
    edge vectors are evaluated.

    Args:
        atomic_numbers (np.ndarray): Per-atom atomic numbers.
        positions (np.ndarray): Cartesian positions, shape (n, 3).
        cell (np.ndarray): 3x3 cell matrix.
        pbc (np.ndarray): Periodic flags.
        edge_index (np.ndarray): Pair indices, shape ``(2, n_pairs)``. Row 0 contains
            receivers and row 1 contains senders.
        unit_shifts (np.ndarray): Integer translations applied to the sender, with shape
            ``(n_pairs, 3)``.
        atom_types (np.ndarray or None): Zero-based element indices, with shape
            ``(n_atoms,)``, or None when atom types are unavailable.
        energy (float, optional): Reference total energy. Any desired energy offset must
            already have been subtracted. Default: None.
        forces (np.ndarray, optional): Reference forces, with shape ``(n_atoms, 3)``.
            Default: None.
        stress (np.ndarray, optional): Reference stress, with shape ``(3, 3)`` or
            ``(6,)`` in ASE Voigt order. Stored as a tensor of shape ``(1, 6)``,
            together with ``virials = -stress * volume``. Default: None.
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
        edge_index: np.ndarray,
        unit_shifts: np.ndarray,
        atom_types: Optional[np.ndarray],
        energy: Optional[float] = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[np.ndarray] = None,
    ):
        dtype = torch.get_default_dtype()
        n_atoms = len(positions)

        cell_t = torch.as_tensor(cell, dtype=dtype)
        if cell_t.shape == (3, 3):
            cell_t = cell_t.unsqueeze(0)

        pbc_t = torch.as_tensor(pbc, dtype=torch.bool)
        if pbc_t.ndim == 0:
            pbc_t = pbc_t.expand(3)
        if pbc_t.shape == (3,):
            pbc_t = pbc_t.unsqueeze(0)

        stress_t = None
        virials_t = None
        if stress is not None:
            if stress.shape == (3, 3):
                from flashcart.utils.geometry import to_voigt6

                stress = to_voigt6(stress)
            stress_t = torch.as_tensor(stress, dtype=dtype).unsqueeze(0)
            volume = torch.abs(torch.linalg.det(cell_t.squeeze(0)))
            virials_t = -stress_t * volume

        super().__init__(
            num_nodes=n_atoms,
            n_atoms=torch.tensor([n_atoms], dtype=torch.long),
            atomic_numbers=torch.as_tensor(atomic_numbers, dtype=torch.long),
            positions=torch.as_tensor(positions, dtype=dtype),
            cell=cell_t,
            pbc=pbc_t,
            use_shifts=uses_periodic_shifts(pbc_t),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            shifts=torch.as_tensor(unit_shifts, dtype=torch.long),
            atom_types=torch.as_tensor(atom_types, dtype=torch.long) if atom_types is not None else None,
            energy=torch.tensor([energy], dtype=dtype) if energy is not None else None,
            forces=torch.as_tensor(forces, dtype=dtype) if forces is not None else None,
            stress=stress_t,
            virials=virials_t,
        )

    @classmethod
    def from_config(
        cls,
        config: AtomicConfig,
        elements: List[Union[str, int]],
        energy_offset: float = 0.0,
    ) -> "AtomicData":
        """Convert an atomic configuration to a tensor graph.

        If no neighbor list is attached, the graph has no edges. An attached neighbor
        list must already have been built. The reference energy is reduced by
        ``energy_offset`` when it is present.

        Args:
            config (AtomicConfig): Source structure.
            elements (list[str | int]): Element order defining atom types.
            energy_offset (float, optional): Subtracted from the reference energy (the
                summed per-element shifts). Default: 0.0.

        Returns:
            AtomicData: Graph containing the geometry, connectivity, atom types, and
                available reference properties.
        """
        atom_types = numbers_to_types(config.atomic_numbers, elements)
        nl = config.neighbors
        energy = config.energy
        if energy is not None and energy_offset != 0.0:
            energy = float(energy) - float(energy_offset)
        data = cls(
            atomic_numbers=config.atomic_numbers,
            positions=config.positions,
            cell=config.cell,
            pbc=config.pbc,
            edge_index=nl.edge_index if nl is not None else np.empty((2, 0), dtype=np.int64),
            unit_shifts=nl.unit_shifts if nl is not None else np.empty((0, 3), dtype=np.int32),
            atom_types=atom_types,
            energy=energy,
            forces=config.forces,
            stress=config.stress,
        )
        data._r_max = nl.r_max if nl is not None else float("nan")
        data._skin = nl.skin if nl is not None else 0.0
        return data

    def to_config(self) -> "AtomicConfig":
        """Convert a single graph to a NumPy configuration.

        Tensor fields are detached and transferred to CPU. Connectivity and integer
        periodic shifts are retained when present. The stored energy is copied without
        restoring any offset subtracted by ``from_config``.

        Returns:
            AtomicConfig: Configuration containing the graph's geometry, available
                reference properties, and neighbor list.
        """
        config = AtomicConfig(
            atomic_numbers=self.atomic_numbers.detach().cpu().numpy(),
            positions=self.positions.detach().cpu().numpy(),
            cell=self.cell.squeeze(0).detach().cpu().numpy().astype(np.float64),
            pbc=self.pbc.squeeze(0).detach().cpu().numpy().astype(bool),
            energy=float(self.energy.squeeze(0).item()) if self.energy is not None else None,
            forces=self.forces.detach().cpu().numpy() if self.forces is not None else None,
            stress=self.stress.squeeze(0).detach().cpu().numpy() if self.stress is not None else None,
        )
        if self.edge_index is not None and self.shifts is not None:
            config.neighbors = NeighborList.from_arrays(
                edge_index=self.edge_index.detach().cpu().numpy(),
                unit_shifts=self.shifts.detach().cpu().numpy(),
                r_max=getattr(self, "_r_max", float("nan")),
                skin=getattr(self, "_skin", 0.0),
            )
        return config
