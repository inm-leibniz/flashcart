import pytest
import torch

from flashcart.o3._linear import _NUM_L_SLOTS, TRITON_AVAILABLE
from flashcart.o3.linear import LinearLayer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="Triton linear tests require CUDA and Triton",
)


@pytest.fixture(autouse=True)
def _ieee_matmul_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _copy_linear_weights(src: LinearLayer, dst: LinearLayer) -> None:
    with torch.no_grad():
        for s, d in zip(src.weight, dst.weight, strict=True):
            d.copy_(s)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "l_max",
    [
        0,
        2,
        pytest.param(4, marks=pytest.mark.skipif(_NUM_L_SLOTS < 5, reason="requires FLASHCART_KERNEL_L_MAX >= 4")),
    ],
)
def test_triton_matches_py_forward_backward(l_max: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    tol = {"atol": 3e-3, "rtol": 2e-3} if dtype == torch.float32 else {"atol": 1e-10, "rtol": 1e-10}
    py_layer = LinearLayer(l_max, l_max, 16, 20, use_triton=False).to(device=device, dtype=dtype)
    triton_layer = LinearLayer(l_max, l_max, 16, 20, use_triton=True).to(device=device, dtype=dtype)
    _copy_linear_weights(py_layer, triton_layer)

    x = torch.randn(6, py_layer.in_dim, device=device, dtype=dtype, requires_grad=True)
    x_py = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(6, py_layer.out_dim, device=device, dtype=dtype)

    y_triton = triton_layer(x)
    y_py = py_layer(x_py)
    torch.testing.assert_close(y_triton, y_py, **tol)

    triton_params = (x, *triton_layer.weight)
    py_params = (x_py, *py_layer.weight)
    grads = torch.autograd.grad(y_triton, triton_params, grad_out)
    py_grads = torch.autograd.grad(y_py, py_params, grad_out)

    torch.testing.assert_close(grads[0], py_grads[0], **tol)
    for g, pg in zip(grads[1:], py_grads[1:]):
        torch.testing.assert_close(g, pg, **tol)


@pytest.mark.parametrize("in_l_max,out_l_max", [(3, 2), (2, 3)])
def test_triton_matches_py_asymmetric_l_max(in_l_max: int, out_l_max: int) -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    dtype = torch.float32
    py_layer = LinearLayer(in_l_max, out_l_max, 16, 20, use_triton=False).to(device=device, dtype=dtype)
    triton_layer = LinearLayer(in_l_max, out_l_max, 16, 20, use_triton=True).to(device=device, dtype=dtype)
    _copy_linear_weights(py_layer, triton_layer)

    x = torch.randn(6, py_layer.in_dim, device=device, dtype=dtype, requires_grad=True)
    x_py = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(6, py_layer.out_dim, device=device, dtype=dtype)

    y_triton = triton_layer(x)
    y_py = py_layer(x_py)
    torch.testing.assert_close(y_triton, y_py, atol=3e-3, rtol=2e-3)

    grads = torch.autograd.grad(y_triton, (x, *triton_layer.weight), grad_out)
    py_grads = torch.autograd.grad(y_py, (x_py, *py_layer.weight), grad_out)
    for g, pg in zip(grads, py_grads):
        torch.testing.assert_close(g, pg, atol=3e-3, rtol=2e-3)


def test_triton_matches_py_wgrad_row_split() -> None:
    from flashcart.o3._linear import _wgrad_split_n

    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.float32
    l_max, n_batch, features = 1, 8192, 32
    py_layer = LinearLayer(l_max, l_max, features, features, use_triton=False).to(device=device, dtype=dtype)
    triton_layer = LinearLayer(l_max, l_max, features, features, use_triton=True).to(device=device, dtype=dtype)
    _copy_linear_weights(py_layer, triton_layer)

    m_values = triton_layer.meta_lists[3]
    f_in_values = triton_layer.meta_lists[4]
    f_out_values = triton_layer.meta_lists[5]
    assert _wgrad_split_n(n_batch, m_values, f_in_values, f_out_values, torch.device("cuda", 0)) > 1

    x = torch.randn(n_batch, py_layer.in_dim, device=device, dtype=dtype, requires_grad=True)
    x_py = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(n_batch, py_layer.out_dim, device=device, dtype=dtype)

    py_grads = torch.autograd.grad(py_layer(x_py), py_layer.weight, grad_out)
    first = torch.autograd.grad(triton_layer(x), triton_layer.weight, grad_out)
    second = torch.autograd.grad(triton_layer(x), triton_layer.weight, grad_out)

    for f, s, pg in zip(first, second, py_grads):
        torch.testing.assert_close(f, pg, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(s, pg, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(f, s, atol=1e-4, rtol=1e-4)


def test_triton_matches_py_double_backward() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float32
    l_max = 3
    py_layer = LinearLayer(l_max, l_max, 12, 14, use_triton=False).to(device=device, dtype=dtype)
    triton_layer = LinearLayer(l_max, l_max, 12, 14, use_triton=True).to(device=device, dtype=dtype)
    _copy_linear_weights(py_layer, triton_layer)

    x = torch.randn(5, py_layer.in_dim, device=device, dtype=dtype, requires_grad=True)
    x_py = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(5, py_layer.out_dim, device=device, dtype=dtype, requires_grad=True)
    grad_out_py = grad_out.detach().clone().requires_grad_(True)

    y_triton = triton_layer(x)
    y_py = py_layer(x_py)

    triton_params = (x, *triton_layer.weight)
    py_params = (x_py, *py_layer.weight)
    grads = torch.autograd.grad(y_triton, triton_params, grad_out, create_graph=True)
    py_grads = torch.autograd.grad(y_py, py_params, grad_out_py, create_graph=True)

    v_x = torch.randn_like(grads[0])
    v_w = [torch.randn_like(g) for g in grads[1:]]
    scalar = (grads[0] * v_x).sum() + sum((g * v).sum() for g, v in zip(grads[1:], v_w))
    py_scalar = (py_grads[0] * v_x).sum() + sum((g * v).sum() for g, v in zip(py_grads[1:], v_w))

    dbwd = torch.autograd.grad(scalar, (grad_out, x, *triton_layer.weight), allow_unused=True)
    py_dbwd = torch.autograd.grad(py_scalar, (grad_out_py, x_py, *py_layer.weight), allow_unused=True)

    torch.testing.assert_close(dbwd[0], py_dbwd[0], atol=5e-3, rtol=3e-3)
    torch.testing.assert_close(dbwd[1], py_dbwd[1], atol=5e-3, rtol=3e-3)
    for d, pd in zip(dbwd[2:], py_dbwd[2:]):
        torch.testing.assert_close(d, pd, atol=5e-3, rtol=3e-3)


def test_triton_matches_py_create_graph_backward() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.float32
    layer = LinearLayer(2, 2, 16, 16, use_triton=True).to(device=device, dtype=dtype)
    x = torch.randn(32, layer.in_dim, device=device, dtype=dtype, requires_grad=True)
    y = layer(x)
    energy = y.square().sum()
    grad_x = torch.autograd.grad(energy, x, create_graph=True)[0]
    loss = grad_x.square().sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
