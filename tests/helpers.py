import sys
import types
from typing import Optional

import numpy as np
import pytest
import scipy.spatial.transform
import torch

from flashcart.data.data import AtomicConfig, AtomicData
from flashcart.model.flashcart import FlashCartPotential
from flashcart.o3._irreps import KERNEL_L_MAX as IRREPS_KERNEL_L_MAX  # noqa: F401  (re-export)
from flashcart.o3._irreps import TRITON_AVAILABLE as _IRREPS_TRITON_AVAILABLE
from flashcart.o3._linear import TRITON_AVAILABLE as _LINEAR_TRITON_AVAILABLE
from flashcart.o3._tensor_product import KERNEL_L_MAX as TP_KERNEL_L_MAX  # noqa: F401  (re-export)
from flashcart.o3._tensor_product import TRITON_AVAILABLE as _TP_TRITON_AVAILABLE
from flashcart.utils.torch_geometric.data import Data

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TRITON_TEST_AVAILABLE = (
    torch.cuda.is_available()
    and _IRREPS_TRITON_AVAILABLE
    and _LINEAR_TRITON_AVAILABLE
    and _TP_TRITON_AVAILABLE
)


def exec_with_triton_stubs(src: str) -> dict:
    tl_stub = types.ModuleType("triton.language")
    tl_stub.__getattr__ = lambda name: None
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda f: f
    triton_stub.autotune = lambda *args, **kwargs: (lambda f: f)
    triton_stub.Config = lambda *args, **kwargs: None
    triton_stub.language = tl_stub

    sentinel = object()
    saved = {name: sys.modules.get(name, sentinel) for name in ("triton", "triton.language")}
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = tl_stub
    try:
        ns: dict = {}
        exec(src, ns, ns)
    finally:
        for name, mod in saved.items():
            if mod is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return ns


def random_rotation(
    seed: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return torch.as_tensor(
        scipy.spatial.transform.Rotation.random(random_state=seed).as_matrix(),
        dtype=dtype if dtype is not None else torch.get_default_dtype(),
        device=device,
    )


def tiny_graph(positions: torch.Tensor, atom_types: torch.Tensor, r_max: float) -> Data:
    n = positions.shape[0]
    device = positions.device
    distances = (positions[:, None, :] - positions[None, :, :]).norm(dim=-1)
    dst, src = torch.where((distances > 1.0e-8) & (distances < r_max))
    edge_index = torch.stack([dst, src], dim=0)
    return Data(
        positions=positions.clone(),
        atom_types=atom_types,
        atomic_numbers=atom_types + 1,
        edge_index=edge_index,
        batch=torch.zeros(n, dtype=torch.long, device=device),
        n_atoms=torch.tensor([n], dtype=torch.long, device=device),
        cell=torch.zeros(3, 3, dtype=positions.dtype, device=device),
        shifts=torch.zeros(edge_index.shape[1], 3, dtype=positions.dtype, device=device),
        pbc=torch.tensor([[False, False, False]], device=device),
        use_shifts=False,
    )


def tiny_periodic_graph(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    elements: list,
    r_max: float,
) -> AtomicData:
    config = AtomicConfig(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=np.array([True, True, True]),
    )
    config.compute_neighbors(r_max)
    data = AtomicData.from_config(config, elements=elements)
    data.batch = torch.zeros(len(positions), dtype=torch.long)
    return data


def batch_graphs(graphs: list) -> Data:
    dtype = graphs[0].positions.dtype
    counts = [g.positions.shape[0] for g in graphs]
    offsets = [sum(counts[:i]) for i in range(len(graphs))]
    n_edges = sum(g.edge_index.shape[1] for g in graphs)
    return Data(
        positions=torch.cat([g.positions for g in graphs]),
        atom_types=torch.cat([g.atom_types for g in graphs]),
        atomic_numbers=torch.cat([g.atomic_numbers for g in graphs]),
        edge_index=torch.cat([g.edge_index + off for g, off in zip(graphs, offsets)], dim=1),
        batch=torch.cat(
            [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)]
        ),
        n_atoms=torch.tensor(counts, dtype=torch.long),
        cell=torch.zeros(len(graphs), 3, 3, dtype=dtype),
        shifts=torch.zeros(n_edges, 3, dtype=dtype),
        pbc=torch.tensor([[False, False, False]] * len(graphs)),
        use_shifts=False,
    )


def tiny_potential(use_triton: bool = False, **overrides) -> FlashCartPotential:
    kwargs = dict(
        elements=["H", "O"],
        r_max=3.0,
        n_hidden_feats=4,
        l_max_hidden_feats=1,
        l_max_edge_attrs=2,
        n_radial=4,
        n_interactions=1,
        correlation=2,
        nonlinearity=True,
        layer_norm=True,
        hidden_radial=[8],
        hidden_readout=[8],
        fit_atomic_shifts=False,
        atomic_shifts=[0.0, 0.0],
        use_triton=use_triton,
    )
    kwargs.update(overrides)
    return FlashCartPotential(**kwargs)


def write_tiny_extxyz(path, n_structures: int = 6) -> None:
    lines: list = []
    for idx in range(n_structures):
        energy = -1.0 - 0.1 * idx
        lines.extend(
            [
                "2",
                f'Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy={energy} pbc="F F F"',
                "H 0.0 0.0 0.0 0.0 0.0 0.0",
                f"H {0.8 + 0.01 * idx} 0.0 0.0 0.0 0.0 0.0",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def tiny_model_cfg(**overrides) -> dict:
    cfg = {
        "elements": ["H"],
        "r_max": 3.0,
        "r_min": 0.0,
        "n_hidden_feats": 2,
        "l_max_hidden_feats": 0,
        "l_max_edge_attrs": 1,
        "n_radial": 2,
        "n_interactions": 1,
        "correlation": 1,
        "element_agnostic_interaction": True,
        "element_agnostic_product": True,
        "nonlinearity": False,
        "layer_norm": False,
        "hidden_radial": [4],
        "hidden_readout": [4],
        "radial_mapping": "cosine",
        "cutoff_poly_order": 6,
        "atomic_shifts": None,
        "fit_atomic_shifts": True,
        "backend": "fast",
        "use_triton": False,
    }
    cfg.update(overrides)
    return cfg
