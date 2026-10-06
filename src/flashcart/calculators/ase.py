from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from flashcart.data.graph import graph_from_ase, update_graph_positions
from flashcart.data.padding import PadAtomicData, slice_padded_outputs
from flashcart.model.flashcart import FlashCartPotential


class FlashCartCalculator(Calculator):
    """Evaluate energies, forces, and stress with a FlashCart model in ASE.

    A positive ``skin`` extends the neighbor-list radius beyond the model cutoff,
    allowing graph connectivity to be reused between evaluations. The cutoff envelope
    removes contributions from edges outside the model cutoff.

    Setting ``compile_mode`` enables compiled prediction and graph padding. Padding
    keeps tensor shapes unchanged while the graph fits within the allocated capacities,
    reducing recompilation as the number of edges varies. The capacities grow when
    required. Positive ``pad_n_atoms`` or ``pad_n_edges`` also enables padding without
    compilation.

    Args:
        checkpoint (Union[str, Path], optional): Inference checkpoint directory.
            Required when ``model`` is not provided. Otherwise ignored.
        model (FlashCartPotential, optional): Model to use for prediction. If set, it
            overrides ``checkpoint``.
        device (Union[str, torch.device], optional): Device used for prediction. If
            None, retain the device of a provided model. Checkpoints are loaded on CPU
            unless a device is specified.
        skin (float, optional): Additional neighbor-list radius. The list is constructed
            at ``r_max + skin``. For a fixed cell, it can be reused while every atom has
            moved less than ``skin / 2`` from its position at construction, provided no
            other atomic properties have changed. Cell changes are checked separately.
            Default: 0.0, which rebuilds the list for each calculation.
        compile_mode (str, optional): ``torch.compile`` mode used for prediction.
            Setting a mode also enables graph padding. Default: None, which evaluates
            the model without compilation.
        compile_fullgraph (bool, optional): Whether compilation requires a single graph
            without graph breaks. Used when ``compile_mode`` is set. Default: True.
        pad_n_atoms (int, optional): Initial capacity for atoms before adding reserved
            padding atoms. The capacity grows when required. Default: 0.
        pad_n_edges (int, optional): Initial capacity for edges before rounding to
            ``pad_edge_multiple``. The capacity grows when required. Default: 0.
        pad_extra_atoms (int, optional): Number of additional padding atoms reserved as
            endpoints of padding edges, with a minimum of two. Default: None, which
            determines the number from the atom capacity and ``pad_edge_headroom``.
        pad_atom_multiple (int, optional): Round the total number of atoms, including
            reserved padding atoms, up to this multiple. Default: 1.
        pad_edge_headroom (float, optional): Multiplicative factor applied to the
            required edge count when increasing the edge capacity. Default: 1.1.
        pad_edge_multiple (int, optional): Round the total number of edges, including
            padding edges, up to this multiple. Default: 128.
        add_atomic_offsets (bool, optional): Restore the per-element energy shifts
            subtracted from the reference energies during training. These shifts affect
            the returned energy but not forces or stress. Default: False.
        **kwargs: Additional keyword arguments passed to ``ase.calculators.Calculator``.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        model: Optional[FlashCartPotential] = None,
        device: Optional[Union[str, torch.device]] = None,
        skin: float = 0.0,
        compile_mode: Optional[str] = None,
        compile_fullgraph: bool = True,
        pad_n_atoms: int = 0,
        pad_n_edges: int = 0,
        pad_extra_atoms: Optional[int] = None,
        pad_atom_multiple: int = 1,
        pad_edge_headroom: float = 1.1,
        pad_edge_multiple: int = 128,
        add_atomic_offsets: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model is None:
            if checkpoint is None:
                raise ValueError("Provide either checkpoint or model.")
            model = FlashCartPotential.from_checkpoint(checkpoint, device=device)
        elif device is not None:
            model = model.to(device)
        self.model = model
        self._device = next(model.parameters()).device
        self.add_atomic_offsets = bool(add_atomic_offsets)
        self._atomic_shifts = model.scale_shift.shifts.detach().to(self._device)
        self.skin = float(skin)
        self.compile_mode = compile_mode
        self.compile_fullgraph = bool(compile_fullgraph)
        self._use_compile = compile_mode is not None
        self._use_padding = self._use_compile or pad_n_atoms > 0 or pad_n_edges > 0
        self._padder = PadAtomicData(
            r_max=model.r_max,
            atom_budget=pad_n_atoms,
            edge_budget=pad_n_edges,
            extra_atoms=pad_extra_atoms,
            atom_multiple=pad_atom_multiple,
            edge_headroom=pad_edge_headroom,
            edge_multiple=pad_edge_multiple,
        )
        self._cached_graph = None
        self._cached_positions: Optional[np.ndarray] = None
        self._cached_cell: Optional[np.ndarray] = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        skin: float = 0.0,
        **kwargs,
    ) -> "FlashCartCalculator":
        """Build a calculator from a saved inference checkpoint.

        Args:
            checkpoint (Union[str, Path]): Inference checkpoint directory.
            device (Union[str, torch.device], optional): Device used for prediction.
                Default: None, which loads the model on CPU.
            skin (float, optional): Additional neighbor-list radius, as described in
                ``FlashCartCalculator``. Default: 0.0.
            **kwargs: Additional arguments passed to ``FlashCartCalculator``.

        Returns:
            FlashCartCalculator: Calculator initialized with the saved model.
        """
        return cls(checkpoint=checkpoint, device=device, skin=skin, **kwargs)

    def calculate(self, atoms=None, properties: Optional[List[str]] = None, system_changes=all_changes):
        """Evaluate the requested properties and store them in ``self.results``.

        Graph connectivity is reused when the neighbor-list reuse conditions are
        satisfied. Otherwise, the graph is rebuilt. The results contain only the
        requested properties. Energy is stored as a scalar, forces as an array of shape
        ``(n_atoms, 3)``, and stress as a six-component array in ASE Voigt order:
        ``xx, yy, zz, yz, xz, xy``.

        Args:
            atoms (ase.Atoms, optional): Atomic configuration to evaluate. If None, use
                the configuration already stored by the calculator.
            properties (list[str], optional): Properties to evaluate, chosen from
                ``"energy"``, ``"forces"``, and ``"stress"``. Default:
                ``["energy", "forces"]``.
            system_changes (list[str], optional): Changes reported by ASE since the
                preceding evaluation. Changes other than positions or cell require the
                neighbor list to be rebuilt. Default: ``all_changes``.
        """
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None

        graph = self._get_graph(system_changes)
        compute_forces = "forces" in properties
        compute_stress = "stress" in properties
        n_real_atoms = int(graph.positions.shape[0])
        n_real_graphs = 1
        predict_graph = graph
        if self._use_padding:
            predict_graph, n_real_atoms, n_real_graphs = self._padder(graph)

        out = self.model.predict(
            predict_graph,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            use_compile=self._use_compile,
            compile_mode=self.compile_mode or "reduce-overhead",
            fullgraph=self.compile_fullgraph,
            dynamic=False,
        )
        if self._use_padding:
            out = slice_padded_outputs(out, n_real_atoms, n_real_graphs)

        self.results = {}
        if "energy" in properties:
            energy = float(out["energy"].detach().cpu().reshape(-1)[0])
            if self.add_atomic_offsets:
                energy += float(self._atomic_shifts[graph.atom_types].double().sum().cpu())
            self.results["energy"] = energy
        if "forces" in properties:
            self.results["forces"] = out["forces"].detach().cpu().numpy()
        if "stress" in properties:
            stress = out["stress"].detach().cpu().numpy()
            self.results["stress"] = stress.reshape(-1) if stress.size == 6 else full_3x3_to_voigt_6_stress(stress)

    def _get_graph(self, system_changes: List[str]):
        """Build an atomic graph or update the geometry of a cached graph.

        Reuse requires a positive skin and changes limited to positions and cell. Each
        atomic displacement and the Frobenius norm of the cell change, measured from the
        last graph construction, must be less than ``skin / 2``. Reuse updates positions
        and cell while retaining the cached connectivity and integer periodic shifts.

        Args:
            system_changes (list[str]): Changes reported by ASE since the preceding
                evaluation.

        Returns:
            AtomicData: Graph for the current atomic configuration.
        """
        reuse = (
            self.skin > 0.0
            and self._cached_graph is not None
            and self._cached_positions is not None
            and self._cached_cell is not None
            and set(system_changes) <= {"positions", "cell"}
        )
        if reuse:
            disp = np.linalg.norm(self.atoms.get_positions() - self._cached_positions, axis=-1)
            cell_disp = np.linalg.norm(np.asarray(self.atoms.get_cell()) - self._cached_cell)
            if np.all(disp < self.skin / 2) and cell_disp < self.skin / 2:
                return update_graph_positions(self._cached_graph, self.atoms)

        graph = graph_from_ase(
            self.atoms,
            self.model.elements,
            self.model.r_max,
            skin=self.skin,
            device=self._device,
        )
        self._cached_graph = graph
        self._cached_positions = self.atoms.get_positions().copy()
        self._cached_cell = np.asarray(self.atoms.get_cell(), dtype=np.float64).copy()
        return graph
