import pytest
import torch

from helpers import random_rotation, requires_cuda
from flashcart import o3
from flashcart.o3._irreps import KERNEL_L_MAX
from flashcart.o3.utils import rotate_irreps

pytestmark = pytest.mark.usefixtures("fp64_default")

_L4 = pytest.param(
    4,
    marks=pytest.mark.skipif(KERNEL_L_MAX < 4, reason="opt in with FLASHCART_KERNEL_L_MAX=4"),
)

_BATCH = 4


def _make_x(device, seed):
    torch.manual_seed(seed)
    return torch.randn(_BATCH, 3, device=device)


@pytest.mark.parametrize("l_max", [1, 2, 3, _L4])
def test_gradcheck_py(l_max):
    irreps = o3.Irreps(l_max=l_max, use_triton=False)
    x = _make_x("cpu", 0).requires_grad_(True)
    assert torch.autograd.gradcheck(irreps, (x,))
    assert torch.autograd.gradgradcheck(irreps, (x,))


def _parity_setup(l_max, seed):
    device = "cuda"
    py_ir = o3.Irreps(l_max=l_max, use_triton=False).to(device)
    tr_ir = o3.Irreps(l_max=l_max, use_triton=True).to(device)
    return py_ir, tr_ir, _make_x(device, seed)


@requires_cuda
@pytest.mark.parametrize("l_max", [0, 1, 2, 3, _L4])
def test_triton_matches_py_forward(l_max):
    for seed in [0, 1]:
        py_ir, tr_ir, x = _parity_setup(l_max, seed)
        with torch.no_grad():
            py_out = py_ir(x)
            tr_out = tr_ir(x)
        assert py_out.shape == tr_out.shape
        assert (py_out - tr_out).abs().max() < 1e-12, f"Forward differs: {(py_out - tr_out).abs().max():.6e}."


@requires_cuda
@pytest.mark.parametrize("l_max", [1, 2, 3, _L4])
def test_triton_matches_py_backward(l_max):
    py_ir, tr_ir, x = _parity_setup(l_max, 0)
    torch.manual_seed(99)
    grad_out = torch.randn(_BATCH, (l_max + 1) ** 2, device=x.device)

    grads = {}
    for name, ir in [("py", py_ir), ("tr", tr_ir)]:
        x_l = x.detach().clone().requires_grad_(True)
        grads[name] = torch.autograd.grad(ir(x_l), x_l, grad_outputs=grad_out)[0]

    assert (
        grads["py"] - grads["tr"]
    ).abs().max() < 1e-12, f"Gradient differs: {(grads['py'] - grads['tr']).abs().max():.6e}."


@requires_cuda
@pytest.mark.parametrize("l_max", [1, 2, 3, _L4])
def test_triton_matches_py_double_backward(l_max):
    py_ir, tr_ir, x = _parity_setup(l_max, 0)
    torch.manual_seed(199)
    v_base = torch.randn(_BATCH, (l_max + 1) ** 2, device=x.device)
    u = torch.randn_like(x)

    second = {}
    for name, ir in [("py", py_ir), ("tr", tr_ir)]:
        x_l = x.detach().clone().requires_grad_(True)
        v_l = v_base.detach().clone().requires_grad_(True)
        grad1 = torch.autograd.grad(ir(x_l), x_l, grad_outputs=v_l, create_graph=True)[0]
        if not grad1.requires_grad:
            second[name] = None
            continue
        second[name] = torch.autograd.grad(grad1, [x_l, v_l], grad_outputs=u)

    if second["py"] is None:
        assert second["tr"] is None or all(
            g.abs().max() < 1e-12 for g in second["tr"]
        ), "Triton second grad must vanish where the PyTorch one is constant."
        return
    for d_py, d_tr in zip(second["py"], second["tr"]):
        assert (d_py - d_tr).abs().max() < 1e-12, f"Second-order gradient differs: {(d_py - d_tr).abs().max():.6e}."


@requires_cuda
@pytest.mark.parametrize("l_max", [2, 3, _L4])
def test_triton_matches_py_hessian(l_max):
    py_ir, tr_ir, x = _parity_setup(l_max, 0)
    torch.manual_seed(99)
    v = torch.randn(_BATCH, (l_max + 1) ** 2, device=x.device)

    def hessian_block(model, wrt_a, wrt_b):
        x_l = x.detach().clone().requires_grad_(True)
        v_l = v.detach().clone().requires_grad_(True)
        leaves = {"x": x_l, "v": v_l}
        g = torch.autograd.grad((model(x_l) * v_l).sum(), leaves[wrt_a], create_graph=True)[0].reshape(-1)
        n_b = leaves[wrt_b].numel()
        rows = []
        for i in range(g.shape[0]):
            if g[i].requires_grad:
                row = torch.autograd.grad(g[i], leaves[wrt_b], retain_graph=True, allow_unused=True)[0]
                row = row.reshape(-1) if row is not None else torch.zeros(n_b, device=x.device, dtype=x.dtype)
            else:
                row = torch.zeros(n_b, device=x.device, dtype=x.dtype)
            rows.append(row)
        return torch.stack(rows)

    for wrt_a, wrt_b in [("x", "x"), ("v", "v"), ("x", "v")]:
        H_py = hessian_block(py_ir, wrt_a, wrt_b)
        H_tr = hessian_block(tr_ir, wrt_a, wrt_b)
        assert (
            H_py - H_tr
        ).abs().max() < 1e-10, f"Hessian d2f/d({wrt_a})d({wrt_b}) differs: {(H_py - H_tr).abs().max():.6e}"


_EQUIV_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _EQUIV_DEVICES)
@pytest.mark.parametrize("l_max", [0, 1, 2, 3, _L4])
def test_rotation_equivariance(l_max, device):
    irreps = o3.Irreps(l_max=l_max, use_triton=device == "cuda").to(device)

    for seed in [0, 1]:
        x = _make_x(device, seed)
        R = random_rotation(seed=seed).to(device)

        out = irreps(x)
        out_rot = irreps(torch.einsum("ij,bj->bi", R, x))
        out_then_rot = rotate_irreps(out, R, l_max)

        assert torch.allclose(
            out_rot, out_then_rot, atol=1e-12
        ), f"Rotation equivariance failed. Max diff: {(out_rot - out_then_rot).abs().max():.6e}"
