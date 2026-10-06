import logging
import math
from typing import Any, Dict, Optional

import torch

from flashcart.utils.geometry import uses_periodic_shifts
from flashcart.utils.torch_geometric.data import Data

log = logging.getLogger(__name__)


class PadAtomicData:
    """Pad atomic graphs to reusable atom, edge, and graph capacities.

    Real atoms, edges, and graphs occupy the leading entries. Padding atoms belong to
    the final graph and lie on a cubic grid with spacing ``2 * r_max``. Padding edges
    connect distinct padding atoms, so their lengths exceed the model cutoff.

    Padding edges are grouped by receiver and distributed across the padding atoms. This
    retains receiver ordering and avoids assigning all padding edges to one receiver in
    the tensor-product kernels. Real edges must already have the ordering required by
    those kernels.

    Capacities grow when an input exceeds them and are rounded to the specified
    multiples. The atom capacity includes extra atoms for padding edges, and the graph
    capacity includes one padding graph. Tensor shapes and storage are reused while
    capacities, device, and dtypes remain unchanged.

    Returned graphs reference reusable buffers. Complete any evaluation and backward
    pass that uses these buffers before the next call.

    Args:
        r_max (float): Model cutoff. Padding atoms sit on a ``2 * r_max`` grid.
        atom_budget (int, optional): Initial atom budget. Default: 0.
        edge_budget (int, optional): Initial edge budget. Default: 0.
        graph_budget (int, optional): Initial graph budget. Default: 0.
        extra_atoms (int, optional): Extra atoms reserved for padding edges, with a
            minimum of two. Default: None, which sets the count from ``atom_budget``
            and ``edge_headroom``.
        atom_headroom (float, optional): Factor applied to the required atom count when
            increasing its capacity. Default: 1.0.
        atom_multiple (int, optional): Multiple to which the total atom count is
            rounded. Default: 1.
        edge_headroom (float, optional): Factor applied to the required edge count when
            increasing its capacity. Default: 1.0.
        edge_multiple (int, optional): Multiple to which the total edge count is
            rounded. Default: 128.
        graph_headroom (float, optional): Factor applied to the required graph count
            when increasing its capacity. Default: 1.0.
        graph_multiple (int, optional): Multiple to which the total graph count is
            rounded. Default: 1.
    """

    def __init__(
        self,
        r_max: float,
        atom_budget: int = 0,
        edge_budget: int = 0,
        graph_budget: int = 0,
        extra_atoms: Optional[int] = None,
        atom_headroom: float = 1.0,
        atom_multiple: int = 1,
        edge_headroom: float = 1.0,
        edge_multiple: int = 128,
        graph_headroom: float = 1.0,
        graph_multiple: int = 1,
    ) -> None:
        self.r_max = float(r_max)
        self.atom_budget = max(int(atom_budget), 0)
        self.edge_budget = max(int(edge_budget), 0)
        self.graph_budget = max(int(graph_budget), 0)
        self._extra_atoms = None if extra_atoms is None else max(int(extra_atoms), 2)
        self.atom_headroom = float(atom_headroom)
        self.atom_multiple = max(int(atom_multiple), 1)
        self.edge_headroom = float(edge_headroom)
        self.edge_multiple = max(int(edge_multiple), 1)
        self.graph_headroom = float(graph_headroom)
        self.graph_multiple = max(int(graph_multiple), 1)
        self._buffers: Dict[str, torch.Tensor] = {}
        self._buffer_signature: Optional[
            tuple[torch.device, torch.dtype, torch.dtype, torch.dtype, torch.dtype, torch.dtype]
        ] = None

    @property
    def extra_atoms(self) -> int:
        """Number of atoms reserved as endpoints of padding edges."""
        if self._extra_atoms is not None:
            return self._extra_atoms
        return max(8, int(self.atom_budget * (self.edge_headroom - 1.0)) // 4)

    @property
    def total_atoms(self) -> int:
        """Total atom capacity, including reserved atoms and rounding."""
        return _round_up(self.atom_budget + self.extra_atoms, self.atom_multiple)

    @property
    def total_edges(self) -> int:
        """Total edge capacity after rounding."""
        return _round_up(self.edge_budget, self.edge_multiple)

    @property
    def total_graphs(self) -> int:
        """Total graph capacity, including the padding graph and rounding."""
        return _round_up(self.graph_budget + 1, self.graph_multiple)

    def seed_budgets(self, atom_budget: int = 0, edge_budget: int = 0, graph_budget: int = 0) -> None:
        """Increase the initial capacities without reducing existing values.

        Args:
            atom_budget (int, optional): Minimum capacity for real atoms. Default: 0.
            edge_budget (int, optional): Minimum capacity for real edges. Default: 0.
            graph_budget (int, optional): Minimum capacity for real graphs. Default: 0.
        """
        old = (self.atom_budget, self.edge_budget, self.graph_budget)
        self.atom_budget = max(self.atom_budget, int(atom_budget))
        self.edge_budget = max(self.edge_budget, int(edge_budget))
        self.graph_budget = max(self.graph_budget, int(graph_budget))
        if old != (self.atom_budget, self.edge_budget, self.graph_budget):
            log.info(
                "Padding budgets seeded from sampler: %d atoms, %d edges, %d graphs",
                self.atom_budget,
                self.edge_budget,
                self.graph_budget,
            )

    def _ensure_budgets(self, n_atoms: int, n_edges: int, n_graphs: int) -> None:
        old = (self.atom_budget, self.edge_budget, self.graph_budget)

        if n_atoms > self.atom_budget:
            self.atom_budget = max(n_atoms, int(n_atoms * self.atom_headroom))
        if n_edges > self.edge_budget:
            self.edge_budget = max(n_edges, int(n_edges * self.edge_headroom))
        if n_graphs > self.graph_budget:
            self.graph_budget = max(n_graphs, int(n_graphs * self.graph_headroom))

        if old != (self.atom_budget, self.edge_budget, self.graph_budget):
            log.info(
                "Padding budgets: %d atoms, %d edges, %d graphs (real: %d atoms, %d edges, %d graphs)",
                self.atom_budget,
                self.edge_budget,
                self.graph_budget,
                n_atoms,
                n_edges,
                n_graphs,
            )

    def _ensure_buffers(
        self,
        device: torch.device,
        dtype: torch.dtype,
        atom_dtype: torch.dtype,
        number_dtype: torch.dtype,
        edge_dtype: torch.dtype,
        shift_dtype: torch.dtype,
    ) -> None:
        total_atoms = self.total_atoms
        total_edges = self.total_edges
        total_graphs = self.total_graphs
        signature = (device, dtype, atom_dtype, number_dtype, edge_dtype, shift_dtype)
        if (
            self._buffers
            and self._buffer_signature == signature
            and self._buffers["positions"].shape[0] == total_atoms
            and self._buffers["edge_index"].shape[1] == total_edges
            and self._buffers["cell"].shape[0] == total_graphs
        ):
            return

        m = max(int(math.ceil(total_atoms ** (1.0 / 3.0))), 1)
        while m**3 < total_atoms:
            m += 1
        idx = torch.arange(total_atoms, device=device)
        grid = torch.stack([idx % m, (idx // m) % m, idx // (m * m)], dim=1)
        spacing = 2.0 * self.r_max

        cell_template = torch.zeros(total_graphs, 3, 3, device=device, dtype=dtype)
        cell_template[:, 0, 0] = cell_template[:, 1, 1] = cell_template[:, 2, 2] = max(spacing, 1.0)

        self._buffer_signature = signature
        self._buffers = {
            "positions": torch.zeros(total_atoms, 3, device=device, dtype=dtype),
            "atom_types": torch.zeros(total_atoms, device=device, dtype=atom_dtype),
            "atomic_numbers": torch.ones(total_atoms, device=device, dtype=number_dtype),
            "batch": torch.zeros(total_atoms, device=device, dtype=torch.long),
            "edge_index": torch.empty(2, total_edges, device=device, dtype=edge_dtype),
            "shifts": torch.zeros(total_edges, 3, device=device, dtype=shift_dtype),
            "cell": torch.zeros(total_graphs, 3, 3, device=device, dtype=dtype),
            "n_atoms": torch.zeros(total_graphs, device=device, dtype=torch.long),
            "graph_mask": torch.zeros(total_graphs, device=device, dtype=dtype),
            "pbc": torch.zeros(total_graphs, 3, device=device, dtype=torch.bool),
            "grid_positions": grid.to(dtype) * spacing,
            "cell_template": cell_template,
            "edge_arange": torch.arange(total_edges, device=device, dtype=edge_dtype),
        }

    def __call__(self, graph: Data) -> tuple[Data, int, int]:
        """Copy graph inputs into the reusable padding buffers.

        Only fields required for model evaluation are retained. Reference energies,
        forces, and stresses are not copied.

        Args:
            graph (Data): Atomic graph or batch to pad.

        Returns:
            tuple[Data, int, int]: Padded graph, number of real atoms, and number of
                real graphs. The graph references reusable storage and must no longer be
                in use when the padder is called again.
        """
        graph_dict = graph.to_dict()
        n_real_atoms = int(graph_dict["positions"].shape[0])
        n_real_edges = int(graph_dict["edge_index"].shape[1])
        n_real_graphs = int(graph_dict["n_atoms"].shape[0])
        self._ensure_budgets(n_real_atoms, n_real_edges, n_real_graphs)

        positions = graph_dict["positions"]
        device = positions.device
        dtype = positions.dtype
        atom_dtype = graph_dict["atom_types"].dtype
        number_dtype = graph_dict["atomic_numbers"].dtype
        edge_dtype = graph_dict["edge_index"].dtype
        shift_dtype = graph_dict["shifts"].dtype

        total_atoms = self.total_atoms
        total_edges = self.total_edges
        total_graphs = self.total_graphs
        junk_graph = total_graphs - 1
        self._ensure_buffers(device, dtype, atom_dtype, number_dtype, edge_dtype, shift_dtype)

        padded_positions = self._buffers["positions"]
        padded_positions.copy_(self._buffers["grid_positions"])
        padded_positions[:n_real_atoms] = positions

        padded_atom_types = self._buffers["atom_types"]
        padded_atom_types.zero_()
        padded_atom_types[:n_real_atoms] = graph_dict["atom_types"]

        padded_atomic_numbers = self._buffers["atomic_numbers"]
        padded_atomic_numbers.fill_(1)
        padded_atomic_numbers[:n_real_atoms] = graph_dict["atomic_numbers"]

        padded_batch = self._buffers["batch"]
        padded_batch.fill_(junk_graph)
        if "batch" in graph_dict and torch.is_tensor(graph_dict["batch"]):
            padded_batch[:n_real_atoms] = graph_dict["batch"]
        else:
            padded_batch[:n_real_atoms] = 0

        padded_edge_index = self._buffers["edge_index"]
        n_fake_edges = total_edges - n_real_edges
        if n_fake_edges > 0:
            n_padding_atoms = total_atoms - n_real_atoms
            chunk = -(-n_fake_edges // n_padding_atoms)
            block = self._buffers["edge_arange"][:n_fake_edges] // chunk
            padded_edge_index[0, n_real_edges:] = n_real_atoms + block
            padded_edge_index[1, n_real_edges:] = n_real_atoms + (block + 1) % n_padding_atoms
        padded_edge_index[:, :n_real_edges] = graph_dict["edge_index"]

        padded_shifts = self._buffers["shifts"]
        padded_shifts.zero_()
        padded_shifts[:n_real_edges] = graph_dict["shifts"]

        padded_cell = self._buffers["cell"]
        padded_cell.copy_(self._buffers["cell_template"])
        padded_cell[:n_real_graphs] = graph_dict["cell"]

        padded_n_atoms = self._buffers["n_atoms"]
        padded_n_atoms.zero_()
        padded_n_atoms[:n_real_graphs] = graph_dict["n_atoms"]
        padded_n_atoms[junk_graph] = total_atoms - n_real_atoms

        padded_graph_mask = self._buffers["graph_mask"]
        padded_graph_mask.zero_()
        padded_graph_mask[:n_real_graphs] = 1.0

        padded: Dict[str, Any] = {
            "positions": padded_positions,
            "cell": padded_cell,
            "shifts": padded_shifts,
            "edge_index": padded_edge_index,
            "batch": padded_batch,
            "atom_types": padded_atom_types,
            "atomic_numbers": padded_atomic_numbers,
            "n_atoms": padded_n_atoms,
            "graph_mask": padded_graph_mask,
            "use_shifts": graph_dict.get("use_shifts", uses_periodic_shifts(graph_dict.get("pbc"))),
        }

        if "pbc" in graph_dict and torch.is_tensor(graph_dict["pbc"]):
            padded_pbc = self._buffers["pbc"]
            padded_pbc.zero_()
            padded_pbc[:n_real_graphs] = graph_dict["pbc"]
            padded["pbc"] = padded_pbc
        return Data.from_dict(padded), n_real_atoms, n_real_graphs


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def slice_padded_outputs(out: Dict[str, Any], n_real_atoms: int, n_real_graphs: int = 1) -> Dict[str, Any]:
    """Remove padding entries from model predictions.

    Args:
        out (dict): Output of ``model.predict`` on a padded graph.
        n_real_atoms (int): Real-atom count returned by the padder.
        n_real_graphs (int, optional): Real-graph count. Default: 1.

    Returns:
        dict[str, Any]: Dictionary containing predictions for real atoms and graphs.
            Tensor slices share storage with the inputs. Unrecognized entries are
            retained unchanged.
    """
    sliced: Dict[str, Any] = {}
    graph_level = {"energy", "stress", "virials"}
    atom_level = {"node_energies", "forces"}
    for key, value in out.items():
        if key in graph_level and torch.is_tensor(value) and value.ndim > 0:
            sliced[key] = value[:n_real_graphs]
        elif key in atom_level and torch.is_tensor(value):
            sliced[key] = value[:n_real_atoms]
        else:
            sliced[key] = value
    return sliced
