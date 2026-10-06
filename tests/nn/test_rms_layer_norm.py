import pytest
import torch

from helpers import random_rotation
from flashcart.nn.layers import RMSLayerNorm
from flashcart.o3.utils import rotate_irreps


def _make_layer(l_max=3, n_features=8, **kwargs) -> RMSLayerNorm:
    defaults = dict(eps=1.0e-12, affine=False, centering=True, use_triton=False)
    defaults.update(kwargs)
    layer = RMSLayerNorm(l_max, n_features, **defaults).to(dtype=torch.float64)
    if defaults["affine"]:
        with torch.no_grad():
            for w in layer.weights:
                w.copy_(torch.randn_like(w))
            if layer.bias is not None:
                layer.bias.copy_(torch.randn_like(layer.bias))
    return layer


_TP_L0_NORMS = (1.0, 1.0, 2.0 / 3.0, 2.0 / 5.0)


def _weighted_mean_square(layer: RMSLayerNorm, feats: torch.Tensor) -> torch.Tensor:
    from flashcart.o3.utils import get_cartesian_slices, irreps_to_cartesian

    n_batch = feats.shape[0]
    n_paths = [1] * (layer.l_max + 1)
    cart_slices = get_cartesian_slices(layer.l_max, 1, n_paths)
    blocks = [
        feats[:, start:stop].view(n_batch, 2 * l + 1, layer.n_features) for l, (start, stop) in enumerate(layer.slices)
    ]
    total = torch.zeros(n_batch, layer.n_features, dtype=feats.dtype)
    for f in range(layer.n_features):
        flat = torch.cat([blocks[l][:, :, f] for l in range(layer.l_max + 1)], dim=-1)
        cart = irreps_to_cartesian(flat, layer.l_max, n_paths)
        for l in range(layer.l_max + 1):
            start, stop = cart_slices[l]
            contraction = cart[:, start:stop].square().sum(dim=-1)
            total[:, f] += layer.l_weights[l] * _TP_L0_NORMS[l] * contraction
    return total.mean(dim=-1)


@pytest.mark.parametrize("affine", [False, True])
def test_forward_rotation_equivariant(affine: bool) -> None:
    torch.manual_seed(1234)
    l_max = 3
    n_features = 8
    layer = _make_layer(affine=affine)

    feats = torch.randn(16, (l_max + 1) ** 2 * n_features, dtype=torch.float64)
    R = random_rotation(seed=1234, dtype=torch.float64)

    normalized_rotated = layer(rotate_irreps(feats, R, l_max))
    rotated_normalized = rotate_irreps(layer(feats), R, l_max)

    assert torch.allclose(
        normalized_rotated, rotated_normalized, atol=1.0e-12, rtol=1.0e-12
    ), f"Max diff: {(normalized_rotated - rotated_normalized).abs().max().item():.6e}"


def test_output_rms_is_unit_without_affine() -> None:
    torch.manual_seed(1234)
    layer = _make_layer(affine=False, centering=False)
    feats = torch.randn(16, (3 + 1) ** 2 * 8, dtype=torch.float64)
    out = layer(feats)
    torch.testing.assert_close(
        _weighted_mean_square(layer, out),
        torch.ones(16, dtype=torch.float64),
        atol=1.0e-9,
        rtol=1.0e-9,
    )


@pytest.mark.parametrize("centering", [False, True])
def test_output_independent_of_input_scale(centering: bool) -> None:
    torch.manual_seed(1234)
    layer = _make_layer(centering=centering)
    feats = torch.randn(16, (3 + 1) ** 2 * 8, dtype=torch.float64)
    torch.testing.assert_close(layer(137.0 * feats), layer(feats), atol=1.0e-9, rtol=1.0e-9)


def test_output_independent_of_l0_constant_offset() -> None:
    torch.manual_seed(1234)
    n_features = 8
    layer = _make_layer(centering=True)
    feats = torch.randn(16, (3 + 1) ** 2 * n_features, dtype=torch.float64)
    shifted = feats.clone()
    shifted[:, :n_features] += torch.randn(16, 1, dtype=torch.float64)
    torch.testing.assert_close(layer(shifted), layer(feats), atol=1.0e-12, rtol=1.0e-12)


def test_gradcheck_py() -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    l_max = 3
    n_features = 3
    n_batch = 4
    layer = RMSLayerNorm(
        l_max,
        n_features,
        eps=1.0e-6,
        affine=True,
        centering=False,
        use_triton=False,
    ).to(dtype=dtype)
    names = [name for name, _ in layer.named_parameters()]
    params = tuple(torch.randn_like(p, requires_grad=True) for p in layer.parameters())
    x = torch.randn(n_batch, (l_max + 1) ** 2 * n_features, dtype=dtype, requires_grad=True)

    def fn(x_in: torch.Tensor, *ws: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(layer, dict(zip(names, ws)), (x_in,))

    assert torch.autograd.gradcheck(fn, (x, *params), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (x, *params), fast_mode=True)
