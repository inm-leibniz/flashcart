from typing import Any, Callable, Optional, Union, Literal

import numpy as np
from ase import Atoms
from matscipy.neighbours import neighbour_list

Backend = Literal["matscipy"]


class NeighborList:
    """Store atom pairs and their integer periodic translations.

    The list is constructed at ``r_max + skin``. The additional radius allows
    connectivity to be reused while it still covers all neighbors within ``r_max``. In
    FlashCart evaluation, the cutoff envelope removes contributions from pairs outside
    the model cutoff.

    Args:
        r_max (float): Cutoff radius.
        skin (float, optional): Extra radius for reuse across steps. Default: 0.0.
        backend (str, optional): Neighbor backend. Default: "matscipy".
    """

    def __init__(self, r_max: float, skin: float = 0.0, backend: Backend = "matscipy"):
        if backend == "matscipy":
            self.impl: Callable = _matscipy_neighbors
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matscipy'.")
        self.r_max = r_max
        self.skin = skin
        self.backend: Backend = backend
        self.edge_index: Optional[np.ndarray] = None
        self.unit_shifts: Optional[np.ndarray] = None

    @property
    def is_built(self) -> bool:
        """Whether pair indices have been assigned."""
        return self.edge_index is not None

    def copy(self) -> "NeighborList":
        """Copy the neighbor list and its arrays.

        Returns:
            NeighborList: Independent copies of the pair-index and shift arrays, or an
                unbuilt list if the original has not been built.
        """
        nl = object.__new__(self.__class__)
        nl.r_max = self.r_max
        nl.skin = self.skin
        nl.backend = self.backend
        nl.impl = self.impl
        nl.edge_index = None if self.edge_index is None else self.edge_index.copy()
        nl.unit_shifts = None if self.unit_shifts is None else self.unit_shifts.copy()
        return nl

    def build(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: Union[bool, list],
    ) -> "NeighborList":
        """Compute ``edge_index`` and ``unit_shifts`` for one structure.

        Args:
            positions (np.ndarray): Cartesian positions, shape (n, 3).
            cell (np.ndarray): 3x3 cell matrix (rows are lattice vectors).
            pbc (bool | list): Periodic flags.

        Returns:
            NeighborList: This instance with ``edge_index`` and ``unit_shifts`` updated.
                Pair-index row 0 contains receivers and row 1 contains senders. Shifts
                translate the senders by integer cell vectors.
        """
        edge_index, unit_shifts = self.impl(positions, cell, pbc, self.r_max, self.skin)
        self.edge_index = np.asarray(edge_index, dtype=np.int64)
        self.unit_shifts = np.asarray(unit_shifts, dtype=np.int32)
        return self

    @classmethod
    def from_arrays(
        cls,
        edge_index: np.ndarray,
        unit_shifts: np.ndarray,
        r_max: float = float("nan"),
        skin: float = 0.0,
        backend: Backend = "matscipy",
    ) -> "NeighborList":
        """Wrap precomputed arrays (e.g. restored from a serialized graph).

        Args:
            edge_index (np.ndarray): Pair indices, shape (2, n_pairs).
            unit_shifts (np.ndarray): Integer image shifts, shape (n_pairs, 3).
            r_max (float, optional): Model cutoff associated with the list. The
                construction radius is ``r_max + skin``. Default: NaN.
            skin (float, optional): Additional radius used when constructing the list.
                Default: 0.0.
            backend (str, optional): Backend to use if the list is rebuilt. Default:
                "matscipy".

        Returns:
            NeighborList: List containing the provided arrays, converted to the required
                integer dtypes. Arrays may share memory with the inputs when conversion
                does not require a copy.
        """
        nl = object.__new__(cls)
        nl.r_max = r_max
        nl.skin = skin
        nl.backend = backend
        if backend == "matscipy":
            nl.impl = _matscipy_neighbors
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'matscipy'.")
        nl.edge_index = np.asarray(edge_index, dtype=np.int64)
        nl.unit_shifts = np.asarray(unit_shifts, dtype=np.int32)
        return nl


# Replace a nearly singular cell with a bounding cell before calling matscipy.
# Remove self-pairs in the reference cell while retaining periodic self-images.
def _matscipy_neighbors(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, list],
    r_max: float,
    skin: float = 0.0,
    eps: float = 1e-7,
    **kwargs: Any,
) -> tuple:
    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)

    if atoms.cell.volume < eps:
        positions_range = np.ptp(positions, axis=0)
        cell_size = np.diag(np.maximum(positions_range, 1e-3) + 2.0 * (r_max + skin))
        atoms.set_cell(cell_size, scale_atoms=False)
        atoms.center()

    sender, receiver, unit_shifts = neighbour_list("ijS", atoms, r_max + skin)

    keep = ~((sender == receiver) & np.all(unit_shifts == 0, axis=1))
    sender, receiver, unit_shifts = sender[keep], receiver[keep], unit_shifts[keep]

    edge_index = np.stack([sender, receiver])
    return edge_index, unit_shifts
