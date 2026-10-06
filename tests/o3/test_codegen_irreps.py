import math

import pytest
import sympy as sp
import torch

from helpers import random_rotation

from flashcart.o3._codegen_common import double_factorial, stored_basis
from flashcart.o3._codegen_irreps import (
    _sym_distinct,
    compile_backward,
    compile_double_backward,
    compile_forward,
    irreps_forward_sympy,
)
from flashcart.o3._irreps import py_irreps
from flashcart.o3.utils import rotate_irreps

pytestmark = pytest.mark.usefixtures("fp64_default")

DTYPE = torch.float64


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def _to_tensor(value, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.full_like(ref, float(value))


def _call_forward(l: int, x: torch.Tensor) -> torch.Tensor:
    fn = compile_forward(l)
    parts = fn(x[:, 0], x[:, 1], x[:, 2])
    return torch.stack([_to_tensor(p, x[:, 0]) for p in parts], dim=-1)


def _full_forward(l_max: int, x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    blocks = [torch.ones(n, 1, dtype=x.dtype, device=x.device)]
    for l in range(1, l_max + 1):
        blocks.append(_call_forward(l, x))
    return torch.cat(blocks, dim=-1)


def _output_dim(l_max: int) -> int:
    return sum(2 * l + 1 for l in range(l_max + 1))


@pytest.mark.parametrize("l_max", [0, 1, 2, 3])
def test_forward_matches_py_irreps(l_max: int):
    torch.manual_seed(0)
    x = _normalize(torch.randn(32, 3, dtype=DTYPE))
    expected = py_irreps(x, l_max)
    got = _full_forward(l_max, x)
    assert got.shape == expected.shape
    diff = (got - expected).abs().max().item()
    assert diff < 1e-12, f"l_max={l_max}: max abs diff {diff:.3e}"


def _call_backward(l: int, x: torch.Tensor, g_l: torch.Tensor) -> torch.Tensor:
    fn = compile_backward(l)
    parts = fn(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        *[g_l[:, i] for i in range(2 * l + 1)],
    )
    gx, gy, gz = parts
    return torch.stack(
        [_to_tensor(gx, x[:, 0]), _to_tensor(gy, x[:, 0]), _to_tensor(gz, x[:, 0])],
        dim=-1,
    )


@pytest.mark.parametrize("l_max", [1, 2, 3, 4])
def test_backward_matches_autograd(l_max: int):
    torch.manual_seed(1)
    x = _normalize(torch.randn(16, 3, dtype=DTYPE))
    x_req = x.detach().clone().requires_grad_(True)

    blocks = [torch.ones(x.shape[0], 1, dtype=DTYPE)]
    for l in range(1, l_max + 1):
        blocks.append(_call_forward(l, x_req))
    full = torch.cat(blocks, dim=-1)

    g_full = torch.randn_like(full)
    (grad_x_auto,) = torch.autograd.grad(full, x_req, g_full, retain_graph=True)

    grad_x_cg = torch.zeros_like(x)
    offset = 1
    for l in range(1, l_max + 1):
        dim_l = 2 * l + 1
        g_l = g_full[:, offset : offset + dim_l]
        grad_x_cg = grad_x_cg + _call_backward(l, x, g_l)
        offset += dim_l

    diff = (grad_x_cg - grad_x_auto).abs().max().item()
    assert diff < 1e-10, f"l_max={l_max}: max abs diff {diff:.3e}"


def _call_double_backward(l: int, x: torch.Tensor, g_l: torch.Tensor, v: torch.Tensor):
    fn = compile_double_backward(l)
    parts = fn(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        *[g_l[:, i] for i in range(2 * l + 1)],
        v[:, 0],
        v[:, 1],
        v[:, 2],
    )
    dgrad_g_components = parts[: 2 * l + 1]
    dx, dy, dz = parts[2 * l + 1 : 2 * l + 4]
    dgrad_g = torch.stack([_to_tensor(c, x[:, 0]) for c in dgrad_g_components], dim=-1)
    dgrad_x = torch.stack(
        [_to_tensor(dx, x[:, 0]), _to_tensor(dy, x[:, 0]), _to_tensor(dz, x[:, 0])],
        dim=-1,
    )
    return dgrad_g, dgrad_x


@pytest.mark.parametrize("l_max", [1, 2, 3, 4])
def test_double_backward_matches_autograd(l_max: int):
    torch.manual_seed(2)
    x = _normalize(torch.randn(8, 3, dtype=DTYPE)).requires_grad_(True)

    blocks = [torch.ones(x.shape[0], 1, dtype=DTYPE)]
    for l in range(1, l_max + 1):
        blocks.append(_call_forward(l, x))
    full = torch.cat(blocks, dim=-1)

    g_full = torch.randn_like(full).requires_grad_(True)
    v = torch.randn_like(x)

    (grad_x_auto,) = torch.autograd.grad(full, x, g_full, create_graph=True)
    loss = (v * grad_x_auto).sum()
    dgrad_x_auto, dgrad_g_auto = torch.autograd.grad(loss, [x, g_full], retain_graph=True, allow_unused=True)
    if dgrad_x_auto is None:
        dgrad_x_auto = torch.zeros_like(x)
    if dgrad_g_auto is None:
        dgrad_g_auto = torch.zeros_like(g_full)

    x_det = x.detach()
    g_det = g_full.detach()
    v_det = v.detach()

    dgrad_g_cg = torch.zeros_like(g_det)
    dgrad_x_cg = torch.zeros_like(x_det)
    offset = 1
    for l in range(1, l_max + 1):
        dim_l = 2 * l + 1
        g_l = g_det[:, offset : offset + dim_l]
        dg, dx_block = _call_double_backward(l, x_det, g_l, v_det)
        dgrad_g_cg[:, offset : offset + dim_l] = dg
        dgrad_x_cg = dgrad_x_cg + dx_block
        offset += dim_l

    err_g = (dgrad_g_cg - dgrad_g_auto).abs().max().item()
    err_x = (dgrad_x_cg - dgrad_x_auto).abs().max().item()
    assert err_g < 1e-10, f"l_max={l_max} dgrad_g diff {err_g:.3e}"
    assert err_x < 1e-10, f"l_max={l_max} dgrad_x diff {err_x:.3e}"


@pytest.mark.parametrize("l_max", [1, 2, 3, 4])
def test_forward_rotation_equivariant(l_max: int):
    torch.manual_seed(7)
    x = _normalize(torch.randn(8, 3, dtype=DTYPE))
    R = random_rotation(seed=7, dtype=DTYPE)

    out = _full_forward(l_max, x)
    out_rot = _full_forward(l_max, torch.einsum("ij,bj->bi", R, x))
    out_then_rot = rotate_irreps(out, R, l_max)

    diff = (out_rot - out_then_rot).abs().max().item()
    assert diff < 1e-12, f"l_max={l_max}: max diff {diff:.3e}"


def _direct_component(l: int, a: int, b: int, c: int) -> sp.Expr:
    x, y, z = sp.symbols("x y z", real=True)
    indices = tuple([0] * a + [1] * b + [2] * c)
    C = sp.Rational(double_factorial(2 * l - 1), math.factorial(l))
    denom = sp.Integer(double_factorial(2 * l - 1))
    expr: sp.Expr = sp.Integer(0)
    for m in range(l // 2 + 1):
        p = l - 2 * m
        prefactor = sp.Integer((-1) ** m * double_factorial(2 * l - 2 * m - 1)) / denom
        tags = {"x": p} if p > 0 else {}
        expr += C * prefactor * _sym_distinct(l, indices, tags, m, {"x": [x, y, z]})
    return sp.expand(expr)


@pytest.mark.parametrize("l", [2, 3, 4])
def test_forward_traceless(l: int):
    components = irreps_forward_sympy(l)
    for idx, (a, b, c) in enumerate(stored_basis(l)):
        assert sp.expand(_direct_component(l, a, b, c) - components[idx]) == 0, (a, b, c)

    x, y, z = sp.symbols("x y z", real=True)
    sphere = x**2 + y**2 + z**2 - 1
    for rest_a in range(l - 1):
        for rest_b in range(l - 1 - rest_a):
            rest_c = l - 2 - rest_a - rest_b
            trace = (
                _direct_component(l, rest_a + 2, rest_b, rest_c)
                + _direct_component(l, rest_a, rest_b + 2, rest_c)
                + _direct_component(l, rest_a, rest_b, rest_c + 2)
            )
            _, remainder = sp.reduced(trace, [sphere], x, y, z)
            assert remainder == 0, f"l={l}: trace over rest=({rest_a},{rest_b},{rest_c}) does not vanish"
