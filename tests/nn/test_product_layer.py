import pytest
import torch

from helpers import random_rotation
from flashcart.nn.layers import ProductLayer
from flashcart.o3.utils import rotate_irreps


@pytest.mark.parametrize(
    ("out_l_max", "correlation", "path_reduced", "n_elements"),
    [
        pytest.param(3, 3, False, None, id="full"),
        pytest.param(3, 3, True, None, id="path_reduced"),
        pytest.param(2, 2, False, None, id="selector_out_lt_in"),
        pytest.param(3, 3, False, 4, id="element_dependent"),
    ],
)
def test_forward_rotation_equivariant(out_l_max: int, correlation: int, path_reduced: bool, n_elements) -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    in_l_max = 3
    n_features = 4
    n_nodes = 6
    R = random_rotation(seed=1234, dtype=dtype)

    layer = ProductLayer(
        in_l_max=in_l_max,
        out_l_max=out_l_max,
        in_features=n_features,
        n_elements=n_elements,
        correlation=correlation,
        path_reduced=path_reduced,
        use_sc=True,
        use_layer_norm=True,
        use_triton=False,
    ).to(dtype=dtype)
    atom_types = None
    if n_elements is not None:
        atom_types = torch.randint(0, n_elements, (n_nodes,))
        # Zero-initialized deltas are invisible; perturb so the element branch
        # actually shapes the output.
        with torch.no_grad():
            layer.path_weights[0].delta.weight.normal_(std=0.1)

    x = torch.randn(n_nodes, layer.in_dim, dtype=dtype)
    out = layer(x, atom_types)
    out_rot = layer(rotate_irreps(x, R, in_l_max), atom_types)
    expected = rotate_irreps(out, R, out_l_max)
    max_diff = (out_rot - expected).abs().max()

    assert torch.allclose(
        out_rot, expected, atol=1.0e-12, rtol=1.0e-12
    ), f"Rotation equivariance failed. Max diff: {max_diff.item():.6e}"


@pytest.mark.parametrize(
    ("path_reduced", "n_elements"),
    [
        pytest.param(False, None, id="full"),
        pytest.param(True, None, id="path_reduced"),
        pytest.param(False, 4, id="element_dependent"),
    ],
)
def test_gradcheck_py(path_reduced: bool, n_elements) -> None:
    torch.manual_seed(1234)
    dtype = torch.float64
    l_max = 3
    n_features = 2
    n_nodes = 3
    layer = ProductLayer(
        in_l_max=l_max,
        out_l_max=l_max,
        in_features=n_features,
        n_elements=n_elements,
        correlation=3,
        path_reduced=path_reduced,
        use_sc=True,
        use_layer_norm=True,
        use_triton=False,
    ).to(dtype=dtype)
    atom_types = None if n_elements is None else torch.randint(0, n_elements, (n_nodes,))
    names = [name for name, _ in layer.named_parameters()]
    params = tuple(torch.randn_like(p, requires_grad=True) for p in layer.parameters())
    x = torch.randn(n_nodes, layer.in_dim, dtype=dtype, requires_grad=True)

    def fn(x_in: torch.Tensor, *ws: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(layer, dict(zip(names, ws)), (x_in, atom_types))

    assert torch.autograd.gradcheck(fn, (x, *params), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (x, *params), fast_mode=True)
