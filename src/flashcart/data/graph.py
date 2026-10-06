from typing import List, Optional, Union

import ase
import numpy as np
import torch

from flashcart.data.data import AtomicConfig, AtomicData
from flashcart.utils.geometry import uses_periodic_shifts


def graph_from_config(
    config: AtomicConfig,
    elements: List[Union[str, int]],
    skin: float = 0.0,
    device: Optional[torch.device] = None,
) -> AtomicData:
    """Construct a model graph from a configuration with a built neighbor list.

    Args:
        config (AtomicConfig): Configuration whose neighbor list has already been built.
        elements (list[str | int]): Element order defining atom types.
        skin (float, optional): Unused by this function. The graph retains the skin
            recorded in ``config.neighbors``. Default: 0.0.
        device (torch.device, optional): Device to which the graph is moved. Default:
            None, which leaves the graph on CPU.

    Returns:
        AtomicData: Graph for one configuration, with every atom assigned to graph index
            zero.

    Raises:
        ValueError: The configuration has no built neighbor list.
    """
    if config.neighbors is None or not config.neighbors.is_built:
        raise ValueError("config.neighbors must be built before graph_from_config().")
    graph = AtomicData.from_config(config, elements)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    if device is not None:
        graph = graph.to(device)
    return graph


def graph_from_ase(
    atoms: ase.Atoms,
    elements: List[Union[str, int]],
    r_max: float,
    skin: float = 0.0,
    device: Optional[torch.device] = None,
) -> AtomicData:
    """Build a model graph directly from an ase.Atoms object.

    Args:
        atoms (ase.Atoms): Structure to convert.
        elements (list[str | int]): Element order defining atom types.
        r_max (float): Cutoff radius for the neighbor list.
        skin (float, optional): Additional neighbor-list radius. The list is built at
            ``r_max + skin``. Default: 0.0.
        device (torch.device, optional): Device to which the graph is moved. Default:
            None, which leaves the graph on CPU.

    Returns:
        AtomicData: Graph containing the atomic configuration, neighbor list, and
            available reference properties.
    """
    config = AtomicConfig.from_ase(atoms)
    config.compute_neighbors(r_max, skin=skin)
    return graph_from_config(config, elements, skin=skin, device=device)


def update_graph_positions(graph: AtomicData, atoms: ase.Atoms) -> AtomicData:
    """Update graph geometry in place while retaining its connectivity.

    Positions, cell, periodicity flags, and ``use_shifts`` are updated from the ASE
    configuration. Connectivity and integer periodic shifts are unchanged. The caller
    must ensure that the retained neighbor list remains valid for the new geometry.

    Args:
        graph (AtomicData): Graph to update.
        atoms (ase.Atoms): Configuration providing the new geometry.

    Returns:
        AtomicData: The input graph with updated geometry.
    """
    dtype = graph.positions.dtype
    device = graph.positions.device
    graph.positions = torch.as_tensor(atoms.get_positions(), dtype=dtype, device=device)
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    graph.cell = torch.as_tensor(cell, dtype=dtype, device=device).unsqueeze(0)
    pbc = atoms.get_pbc()
    graph.pbc = torch.as_tensor(pbc, dtype=torch.bool, device=device).unsqueeze(0)
    graph.use_shifts = uses_periodic_shifts(pbc)
    return graph
