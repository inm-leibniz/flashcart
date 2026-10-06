import pytest
import torch

from helpers import random_rotation
from flashcart.nn.layers import EquivariantGatedLayer, RescaledSigmoidLayer
from flashcart.o3.utils import rotate_irreps


def _make_layer(gate_activation: str, gate_bias: bool, dtype=torch.float64) -> EquivariantGatedLayer:
    torch.manual_seed(1234)
    layer = EquivariantGatedLayer(3, 4, hidden_features=5, gate_activation=gate_activation, gate_bias=gate_bias).to(
        dtype=dtype
    )
    with torch.no_grad():
        for weight in layer.output_linear.weight:
            weight.normal_(std=0.1)
        layer.gate_linear.weight.normal_(std=0.7)
        if layer.gate_linear.bias is not None:
            layer.gate_linear.bias.normal_(std=2.0)
    return layer


@pytest.mark.parametrize("gate_bias", [True, False])
@pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
def test_forward_rotation_equivariant(gate_activation: str, gate_bias: bool) -> None:
    dtype = torch.float64
    l_max = 3
    layer = _make_layer(gate_activation, gate_bias)
    R = random_rotation(seed=99, dtype=dtype)
    feats = torch.randn(6, layer.in_dim, dtype=dtype)
    out = layer(feats)
    out_rot = layer(rotate_irreps(feats, R, l_max))
    expected = rotate_irreps(out, R, l_max)
    max_diff = (out_rot - expected).abs().max().item()
    assert torch.allclose(
        out_rot, expected, atol=1.0e-12, rtol=1.0e-12
    ), f"{gate_activation}/bias={gate_bias}: max diff {max_diff:.3e}"


def test_gates_invariant_under_rotation() -> None:
    dtype = torch.float64
    l_max = 3
    layer = _make_layer("sigmoid", True)
    R = -random_rotation(seed=7, dtype=dtype)
    feats = torch.randn(6, layer.in_dim, dtype=dtype)

    captured = []
    hook = layer.gate_activation.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    layer(feats)
    layer(rotate_irreps(feats, R, l_max))
    hook.remove()
    assert torch.allclose(captured[0], captured[1], atol=1.0e-12, rtol=1.0e-12)


def test_gradcheck_py() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    l_max = 3
    n_features = 2
    n_hidden_features = 3
    n_batch = 3
    layer = EquivariantGatedLayer(
        l_max,
        n_features,
        hidden_features=n_hidden_features,
        use_triton=False,
    ).to(dtype=dtype)
    names = [name for name, _ in layer.named_parameters()]
    params = tuple(torch.randn_like(p, requires_grad=True) for p in layer.parameters())
    x = torch.randn(n_batch, layer.in_dim, dtype=dtype, requires_grad=True)

    def fn(x_in: torch.Tensor, *ws: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(layer, dict(zip(names, ws)), (x_in,))

    assert torch.autograd.gradcheck(fn, (x, *params), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (x, *params), fast_mode=True)
