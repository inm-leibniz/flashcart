import pytest
import torch

from helpers import random_rotation
from flashcart.nn.layers import InteractionLayer
from flashcart.o3._tensor_product import TRITON_AVAILABLE as TP_TRITON_AVAILABLE
from flashcart.o3.utils import rotate_irreps


def test_forward_rotation_equivariant() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    device = torch.device("cpu")
    use_triton = False
    tol = 1.0e-12
    l_max = 3
    n_features = 4
    n_radial = 4
    n_nodes = 6
    layer = InteractionLayer(
        in1_l_max=l_max,
        in2_l_max=l_max,
        out_l_max=l_max,
        in1_features=n_features,
        out_features=n_features,
        n_radial=n_radial,
        hidden_radial=[8],
        avg_neighbors=1.0,
        use_sc=True,
        use_layer_norm=True,
        use_triton=use_triton,
    ).to(device=device, dtype=dtype)

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    node_feats = torch.randn(n_nodes, layer.tp.in1_dim, dtype=dtype, device=device)
    edge_attrs = torch.randn(edge_index.shape[1], layer.tp.in2_dim, dtype=dtype, device=device)
    edge_feats = torch.randn(edge_index.shape[1], n_radial, dtype=dtype, device=device)
    envelope = torch.rand(edge_index.shape[1], dtype=dtype, device=device)
    R = random_rotation(seed=1234, dtype=dtype, device=device)

    out = layer(node_feats, edge_attrs, edge_feats, edge_index, envelope)
    out_rot = layer(
        rotate_irreps(node_feats, R, l_max),
        rotate_irreps(edge_attrs, R, l_max),
        edge_feats,
        edge_index,
        envelope,
    )

    expected = rotate_irreps(out, R, l_max)
    max_diff = (out_rot - expected).abs().max()
    assert torch.allclose(
        out_rot, expected, atol=tol, rtol=tol
    ), f"Rotation equivariance failed. Max diff: {max_diff.item():.6e}"


def test_gradcheck_py() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    l_max = 3
    n_features = 2
    n_radial = 2
    n_nodes = 3
    layer = InteractionLayer(
        in1_l_max=l_max,
        in2_l_max=l_max,
        out_l_max=l_max,
        in1_features=n_features,
        out_features=n_features,
        n_radial=n_radial,
        hidden_radial=[3],
        avg_neighbors=1.0,
        use_sc=True,
        use_layer_norm=True,
        use_triton=False,
    ).to(dtype=dtype)
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
    n_edges = edge_index.shape[1]
    names = [name for name, _ in layer.named_parameters()]
    params = tuple(torch.randn_like(p, requires_grad=True) for p in layer.parameters())
    node_feats = torch.randn(n_nodes, layer.tp.in1_dim, dtype=dtype, requires_grad=True)
    edge_attrs = torch.randn(n_edges, layer.tp.in2_dim, dtype=dtype, requires_grad=True)
    edge_feats = torch.randn(n_edges, n_radial, dtype=dtype, requires_grad=True)
    envelope = torch.rand(n_edges, dtype=dtype, requires_grad=True)

    def fn(nf: torch.Tensor, ea: torch.Tensor, ef: torch.Tensor, env: torch.Tensor, *ws: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(layer, dict(zip(names, ws)), (nf, ea, ef, edge_index, env))

    assert torch.autograd.gradcheck(fn, (node_feats, edge_attrs, edge_feats, envelope, *params), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (node_feats, edge_attrs, edge_feats, envelope, *params), fast_mode=True)


@pytest.mark.parametrize(
    "device_name",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() and TP_TRITON_AVAILABLE else []),
)
def test_recompute_radial_matches_standard(device_name: str) -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    device = torch.device(device_name)
    n_radial, n_nodes = 4, 6

    def build(recompute: bool) -> InteractionLayer:
        torch.manual_seed(99)
        return InteractionLayer(
            in1_l_max=2,
            in2_l_max=3,
            out_l_max=3,
            in1_features=4,
            out_features=4,
            n_radial=n_radial,
            hidden_radial=[8],
            avg_neighbors=2.0,
            use_sc=True,
            use_layer_norm=True,
            use_triton=device.type == "cuda",
            recompute_radial=recompute,
        ).to(device=device, dtype=dtype)

    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 1]],
        dtype=torch.long,
        device=device,
    )
    n_edges = edge_index.shape[1]
    layers = {rc: build(rc) for rc in (False, True)}
    node_feats = torch.randn(n_nodes, layers[False].tp.in1_dim, dtype=dtype, device=device)
    edge_attrs = torch.randn(n_edges, layers[False].tp.in2_dim, dtype=dtype, device=device)
    edge_feats = torch.randn(n_edges, n_radial, dtype=dtype, device=device)
    envelope = torch.rand(n_edges, dtype=dtype, device=device)
    cot = torch.randn(n_nodes, layers[False].out_dim, dtype=dtype, device=device)

    results = {}
    held = {}
    for rc, layer in layers.items():
        params = tuple(layer.parameters())
        inputs = tuple(t.clone().requires_grad_(True) for t in (node_feats, edge_attrs, edge_feats, envelope))
        if device.type == "cuda":
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
        out = layer(inputs[0], inputs[1], inputs[2], edge_index, inputs[3])
        if device.type == "cuda":
            torch.cuda.synchronize()
            held[rc] = torch.cuda.memory_allocated() - base
        first = torch.autograd.grad((out * cot).sum(), (*inputs, *params), create_graph=True)
        total = sum(g.square().sum() for g in first[: len(inputs)])
        second = torch.autograd.grad(total, (*inputs, *params))
        results[rc] = (out, *first, *second)

    for t_std, t_rc in zip(results[False], results[True]):
        assert torch.equal(t_std, t_rc), f"recompute_radial mismatch. Max diff: {(t_std - t_rc).abs().max().item():.6e}"
    if device.type == "cuda":
        assert (
            held[True] < held[False]
        ), f"recompute_radial did not reduce held memory: {held[True]} vs {held[False]} bytes"
