from functools import lru_cache
from itertools import product as iter_product

import pytest
import sympy as sp
import torch

from helpers import exec_with_triton_stubs, random_rotation

from flashcart.o3._codegen_common import stored_basis
from flashcart.o3._codegen_tensor_product import (
    _contract_k_pairs,
    _even_tp_norm,
    _full_cartesian_from_stored,
    _outer_product,
    _stf_projection,
    _sym_average,
    _tp_path_order,
    build_triton_tp_module_source,
    compile_even_tp_backward,
    compile_even_tp_double_backward,
    compile_even_tp_forward,
    even_tp_forward_sympy,
    even_tp_path_allowed,
    even_tp_t1_symbols,
    even_tp_t2_symbols,
)
from flashcart.o3.utils import (
    cartesian_to_irreps,
    get_cartesian_slices,
    get_irreps_slices,
    irreps_to_cartesian,
    rotate_irreps,
)

pytestmark = pytest.mark.usefixtures("fp64_default")

DTYPE = torch.float64


def _pack_irreps_side(
    active_l: int,
    stored: list,
    layout_l_max: int,
    n: int,
    features: int = 1,
) -> torch.Tensor:
    chunks: list = []
    for l in range(layout_l_max + 1):
        dim = 2 * l + 1
        if l == active_l:
            if l == 0:
                chunks.append(stored[0].reshape(n, features))
            else:
                chunks.append(torch.stack(stored, dim=1).reshape(n, dim * features))
        else:
            chunks.append(torch.zeros(n, dim * features, dtype=stored[0].dtype, device=stored[0].device))
    return torch.cat(chunks, dim=1)


def _canonicalize_stored(l_side: int, stored: list, n: int, features: int = 1) -> list:
    if l_side == 0:
        return stored
    flat = _pack_irreps_side(l_side, stored, l_side, n, features)
    n_paths = [1] * (l_side + 1)
    cart = irreps_to_cartesian(flat, l_side, n_paths)
    flat2 = cartesian_to_irreps(cart, l_side, n_paths)
    sl = get_irreps_slices(l_side, features, n_paths)
    block = flat2[:, sl[l_side][0] : sl[l_side][1]]
    return [block[:, i * features : (i + 1) * features].reshape(n) for i in range(2 * l_side + 1)]


def _cartesian_from_stored(l: int, stored: list, n: int) -> torch.Tensor:
    if l == 0:
        return stored[0].reshape(n)
    flat = _pack_irreps_side(l, stored, l, n)
    n_paths = [1] * (l + 1)
    cart = irreps_to_cartesian(flat, l, n_paths)
    sl = get_cartesian_slices(l, 1, n_paths)
    return cart[:, sl[l][0] : sl[l][1]].reshape(n, *([3] * l))


def _stored_from_cartesian(l_out: int, out_full: torch.Tensor, n: int) -> list:
    if l_out == 0:
        return [out_full.reshape(n)]
    parts = []
    for l in range(l_out + 1):
        dim = 3**l
        if l == l_out:
            parts.append(out_full.reshape(n, dim))
        else:
            parts.append(torch.zeros(n, dim, dtype=out_full.dtype, device=out_full.device))
    cart = torch.cat(parts, dim=1)
    n_paths = [1] * (l_out + 1)
    flat = cartesian_to_irreps(cart, l_out, n_paths)
    sl = get_irreps_slices(l_out, 1, n_paths)
    block = flat[:, sl[l_out][0] : sl[l_out][1]]
    return [block[:, i].reshape(n) for i in range(2 * l_out + 1)]


def _einsum_path_contract(l1: int, l2: int, l_out: int, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, dtype=A.dtype if torch.is_tensor(A) else DTYPE)
    key = (l1, l2, l_out)
    if l_out == 0:
        norm = {(0, 0): 1.0, (1, 1): 1.0, (2, 2): 2.0 / 3.0, (3, 3): 2.0 / 5.0}[(l1, l2)]
        subs = {0: "a,a->a", 1: "ai,ai->a", 2: "aij,aij->a", 3: "aijk,aijk->a"}[l1]
        return norm * torch.einsum(subs, A, B)
    if key == (1, 0, 1):
        return torch.einsum("ai,a->ai", A, B)
    if key == (2, 1, 1):
        return torch.einsum("aij,aj->ai", A, B)
    if key == (3, 2, 1):
        return (2.0 / 3.0) * torch.einsum("aijk,ajk->ai", A, B)
    if key == (1, 1, 2):
        xy = torch.einsum("ai,aj->aij", A, B)
        xy = xy + xy.transpose(1, 2)
        z = torch.einsum("ai,ai->a", A, B)
        return (3.0 / 4.0) * (xy - (2.0 / 3.0) * torch.einsum("a,ij->aij", z, eye))
    if key == (2, 0, 2):
        return torch.einsum("aij,a->aij", A, B)
    if key == (2, 2, 2):
        xy = torch.einsum("aij,ajk->aik", A, B)
        xy = xy + xy.transpose(1, 2)
        z = torch.einsum("aij,aij->a", A, B)
        return xy - (2.0 / 3.0) * torch.einsum("a,ij->aij", z, eye)
    if key == (3, 1, 2):
        return torch.einsum("aijk,ak->aij", A, B)
    if key == (3, 3, 2):
        xy = torch.einsum("aijk,ajkl->ail", A, B)
        xy = xy + xy.transpose(1, 2)
        z = torch.einsum("aijk,aijk->a", A, B)
        return (3.0 / 4.0) * (xy - (2.0 / 3.0) * torch.einsum("a,ij->aij", z, eye))
    if key == (2, 1, 3):
        xy = torch.einsum("aij,ak->aijk", A, B)
        xy = xy + xy.permute(0, 2, 3, 1) + xy.permute(0, 3, 1, 2)
        z = torch.einsum("aij,aj->ai", A, B)
        z_eye = torch.einsum("ai,jk->aijk", z, eye)
        z_eye = z_eye + z_eye.permute(0, 2, 3, 1) + z_eye.permute(0, 3, 1, 2)
        return (5.0 / 9.0) * (xy - (2.0 / 5.0) * z_eye)
    if key == (3, 0, 3):
        return torch.einsum("aijk,a->aijk", A, B)
    if key == (3, 2, 3):
        xy = torch.einsum("aijk,akl->aijl", A, B)
        xy = xy + xy.permute(0, 2, 3, 1) + xy.permute(0, 3, 1, 2)
        z = torch.einsum("aijk,ajk->ai", A, B)
        z_eye = torch.einsum("ai,jk->aijk", z, eye)
        z_eye = z_eye + z_eye.permute(0, 2, 3, 1) + z_eye.permute(0, 3, 1, 2)
        return (5.0 / 6.0) * (xy - (2.0 / 5.0) * z_eye)
    raise KeyError(f"no einsum reference for path {key}")


def _ref_contract(l1: int, l2: int, l_out: int, T1: list, T2: list) -> list:
    n = T1[0].shape[0]
    A = _cartesian_from_stored(l1, T1, n)
    B = _cartesian_from_stored(l2, T2, n)
    out_full = _einsum_path_contract(l1, l2, l_out, A, B)
    return _stored_from_cartesian(l_out, out_full, n)


_CANONICAL_PATHS = [
    (0, 0, 0),
    (1, 1, 0),
    (2, 2, 0),
    (3, 3, 0),
    (1, 0, 1),
    (2, 1, 1),
    (3, 2, 1),
    (2, 0, 2),
    (1, 1, 2),
    (2, 2, 2),
    (3, 3, 2),
    (3, 1, 2),
    (3, 0, 3),
    (2, 1, 3),
    (3, 2, 3),
]
for _p in _CANONICAL_PATHS:
    assert even_tp_path_allowed(*_p), f"Unexpected: {_p} not allowed by even-TP selection rule"


@pytest.mark.parametrize("path", _CANONICAL_PATHS, ids=lambda p: f"l{p[0]}_l{p[1]}_to_l{p[2]}")
def test_forward_matches_einsum_reference(path):
    l1, l2, l_out = path
    torch.manual_seed(hash(path) & 0xFFFF)
    n = 4
    T1 = _canonicalize_stored(l1, [torch.randn(n, dtype=DTYPE) for _ in range(2 * l1 + 1)], n)
    T2 = _canonicalize_stored(l2, [torch.randn(n, dtype=DTYPE) for _ in range(2 * l2 + 1)], n)

    fn = compile_even_tp_forward(l1, l2, l_out)
    got = fn(*T1, *T2)
    if not isinstance(got, (list, tuple)):
        got = [got]
    got = [v if torch.is_tensor(v) else torch.full((n,), float(v), dtype=DTYPE) for v in got]

    expected = _ref_contract(l1, l2, l_out, T1, T2)
    expected = [v if torch.is_tensor(v) else torch.full((n,), float(v), dtype=DTYPE) for v in expected]

    assert len(got) == 2 * l_out + 1
    assert len(expected) == 2 * l_out + 1
    for i, (a, b) in enumerate(zip(got, expected)):
        diff = (a - b).abs().max().item()
        assert diff < 1e-11, f"path {path} component {i}: max abs diff {diff:.3e}"


def _rotate_stored(l: int, stored: list, R: torch.Tensor, n: int) -> list:
    if l == 0:
        return list(stored)
    flat = _pack_irreps_side(l, stored, l, n)
    rot = rotate_irreps(flat, R, l)
    sl = get_irreps_slices(l, 1, [1] * (l + 1))
    block = rot[:, sl[l][0] : sl[l][1]]
    return [block[:, i] for i in range(2 * l + 1)]


@pytest.mark.parametrize("path", _tp_path_order(4), ids=lambda p: f"l{p[0]}_l{p[1]}_to_l{p[2]}")
def test_forward_rotation_equivariant(path):
    l1, l2, l_out = path
    torch.manual_seed(37 + hash(path) % 1000)
    n = 4
    R = random_rotation(seed=37, dtype=DTYPE)
    T1 = [torch.randn(n, dtype=DTYPE) for _ in range(2 * l1 + 1)]
    T2 = [torch.randn(n, dtype=DTYPE) for _ in range(2 * l2 + 1)]

    fwd = compile_even_tp_forward(l1, l2, l_out)

    def run(a: list, b: list) -> list:
        out = fwd(*a, *b)
        if not isinstance(out, (list, tuple)):
            out = [out]
        return [v if torch.is_tensor(v) else torch.full((n,), float(v), dtype=DTYPE) for v in out]

    out_then_rot = _rotate_stored(l_out, run(T1, T2), R, n)
    rot_then_out = run(_rotate_stored(l1, T1, R, n), _rotate_stored(l2, T2, R, n))
    for i, (a, b) in enumerate(zip(rot_then_out, out_then_rot)):
        diff = (a - b).abs().max().item()
        assert diff < 1e-10, f"path {path} component {i}: max abs diff {diff:.3e}"


def _full_output_tensor(l1: int, l2: int, l_out: int) -> dict:
    S1 = even_tp_t1_symbols(l1)
    S2 = even_tp_t2_symbols(l2)
    T1 = _full_cartesian_from_stored(l1, S1)
    T2 = _full_cartesian_from_stored(l2, S2)
    F = _outer_product(T1, l1, T2, l2)
    k = (l1 + l2 - l_out) // 2
    C = _contract_k_pairs(F, l1, l2, k)
    return _stf_projection(_sym_average(C, l_out), l_out)


_STF_PATHS = [p for p in _tp_path_order(4) if p[2] >= 2]


@pytest.mark.parametrize("path", _STF_PATHS, ids=lambda p: f"l{p[0]}_l{p[1]}_to_l{p[2]}")
def test_forward_traceless(path):
    l1, l2, l_out = path
    full = _full_output_tensor(l1, l2, l_out)

    norm = _even_tp_norm(l1, l2, l_out)
    emitted = even_tp_forward_sympy(l1, l2, l_out)
    for i, (a, b, c) in enumerate(stored_basis(l_out)):
        idx = tuple([0] * a + [1] * b + [2] * c)
        assert sp.expand(norm * full[idx] - emitted[i]) == 0, f"stored ({a},{b},{c})"

    for idx in full:
        assert sp.expand(full[idx] - full[tuple(sorted(idx))]) == 0, idx

    for rest in iter_product(range(3), repeat=l_out - 2):
        trace = sum(full[(a, a) + rest] for a in range(3))
        assert sp.expand(trace) == 0, f"trace over rest={rest} does not vanish"


@pytest.mark.parametrize("path", _CANONICAL_PATHS, ids=lambda p: f"l{p[0]}_l{p[1]}_to_l{p[2]}")
def test_backward_matches_autograd(path):
    l1, l2, l_out = path
    torch.manual_seed(13 + hash(path) % 1000)
    n = 4
    n1, n2, n_out = 2 * l1 + 1, 2 * l2 + 1, 2 * l_out + 1

    T1 = [torch.randn(n, dtype=DTYPE, requires_grad=True) for _ in range(n1)]
    T2 = [torch.randn(n, dtype=DTYPE, requires_grad=True) for _ in range(n2)]
    g = [torch.randn(n, dtype=DTYPE) for _ in range(n_out)]

    fwd = compile_even_tp_forward(l1, l2, l_out)
    parts = fwd(*T1, *T2)
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    parts = [p if torch.is_tensor(p) else torch.full((n,), float(p), dtype=DTYPE) for p in parts]
    loss = sum((g[a] * parts[a]).sum() for a in range(n_out))

    grads = torch.autograd.grad(loss, T1 + T2, allow_unused=True)
    grad_T1_auto = [gg if gg is not None else torch.zeros(n, dtype=DTYPE) for gg in grads[:n1]]
    grad_T2_auto = [gg if gg is not None else torch.zeros(n, dtype=DTYPE) for gg in grads[n1:]]

    bwd = compile_even_tp_backward(l1, l2, l_out)
    T1_det = [t.detach() for t in T1]
    T2_det = [t.detach() for t in T2]
    parts = bwd(*T1_det, *T2_det, *g)
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    parts = [p if torch.is_tensor(p) else torch.full((n,), float(p), dtype=DTYPE) for p in parts]
    grad_T1_cg, grad_T2_cg = parts[:n1], parts[n1:]

    for i in range(n1):
        diff = (grad_T1_cg[i] - grad_T1_auto[i]).abs().max().item()
        assert diff < 1e-10, f"path {path} grad_T1[{i}] diff {diff:.3e}"
    for i in range(n2):
        diff = (grad_T2_cg[i] - grad_T2_auto[i]).abs().max().item()
        assert diff < 1e-10, f"path {path} grad_T2[{i}] diff {diff:.3e}"


@pytest.mark.parametrize("path", _CANONICAL_PATHS, ids=lambda p: f"l{p[0]}_l{p[1]}_to_l{p[2]}")
def test_double_backward_matches_autograd(path):
    l1, l2, l_out = path
    torch.manual_seed(101 + hash(path) % 1000)
    n = 3
    n1, n2, n_out = 2 * l1 + 1, 2 * l2 + 1, 2 * l_out + 1

    T1 = [torch.randn(n, dtype=DTYPE, requires_grad=True) for _ in range(n1)]
    T2 = [torch.randn(n, dtype=DTYPE, requires_grad=True) for _ in range(n2)]
    g = [torch.randn(n, dtype=DTYPE, requires_grad=True) for _ in range(n_out)]
    vT1 = [torch.randn(n, dtype=DTYPE) for _ in range(n1)]
    vT2 = [torch.randn(n, dtype=DTYPE) for _ in range(n2)]

    fwd = compile_even_tp_forward(l1, l2, l_out)
    parts = fwd(*T1, *T2)
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    parts = [p if torch.is_tensor(p) else torch.full((n,), float(p), dtype=DTYPE) for p in parts]
    loss1 = sum((g[a] * parts[a]).sum() for a in range(n_out))
    grads = torch.autograd.grad(loss1, T1 + T2, create_graph=True, allow_unused=True)
    grads = [gg if gg is not None else torch.zeros(n, dtype=DTYPE) for gg in grads]
    grad_T1, grad_T2 = grads[:n1], grads[n1:]

    loss2 = sum((vT1[b] * grad_T1[b]).sum() for b in range(n1)) + sum((vT2[b] * grad_T2[b]).sum() for b in range(n2))

    d_inputs = list(T1) + list(T2) + list(g)
    d_outputs = torch.autograd.grad(loss2, d_inputs, retain_graph=True, allow_unused=True)
    n_in = n1 + n2
    dgrad_T1_auto = [d if d is not None else torch.zeros(n, dtype=DTYPE) for d in d_outputs[:n1]]
    dgrad_T2_auto = [d if d is not None else torch.zeros(n, dtype=DTYPE) for d in d_outputs[n1:n_in]]
    dgrad_g_auto = [d if d is not None else torch.zeros(n, dtype=DTYPE) for d in d_outputs[n_in:]]

    dbwd = compile_even_tp_double_backward(l1, l2, l_out)
    T1_det = [t.detach() for t in T1]
    T2_det = [t.detach() for t in T2]
    g_det = [t.detach() for t in g]
    parts = dbwd(*T1_det, *T2_det, *g_det, *vT1, *vT2)
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    parts = [p if torch.is_tensor(p) else torch.full((n,), float(p), dtype=DTYPE) for p in parts]
    dgrad_g_cg = parts[:n_out]
    dgrad_T1_cg = parts[n_out : n_out + n1]
    dgrad_T2_cg = parts[n_out + n1 : n_out + n1 + n2]

    for i in range(n_out):
        diff = (dgrad_g_cg[i] - dgrad_g_auto[i]).abs().max().item()
        assert diff < 1e-10, f"path {path} dgrad_g[{i}] diff {diff:.3e}"
    for i in range(n1):
        diff = (dgrad_T1_cg[i] - dgrad_T1_auto[i]).abs().max().item()
        assert diff < 1e-10, f"path {path} dgrad_T1[{i}] diff {diff:.3e}"
    for i in range(n2):
        diff = (dgrad_T2_cg[i] - dgrad_T2_auto[i]).abs().max().item()
        assert diff < 1e-10, f"path {path} dgrad_T2[{i}] diff {diff:.3e}"


@lru_cache(maxsize=None)
def _exec_generated_module(kernel_l_max: int):
    return exec_with_triton_stubs(build_triton_tp_module_source(kernel_l_max))


_ALL_PATHS_L4 = _tp_path_order(4)


@pytest.mark.parametrize("path", _ALL_PATHS_L4, ids=lambda p: f"l{p[0]}_l{p[1]}_to_l{p[2]}")
def test_emitted_contract_matches_sympy(path):
    l1, l2, l_out = path
    ns = _exec_generated_module(4)
    fn = ns[f"contract_l{l1}_l{l2}_to_l{l_out}"]
    fwd = compile_even_tp_forward(l1, l2, l_out)

    torch.manual_seed(17 + hash(path) % 1000)
    n = 5
    T1 = [torch.randn(n, dtype=DTYPE) for _ in range(2 * l1 + 1)]
    T2 = [torch.randn(n, dtype=DTYPE) for _ in range(2 * l2 + 1)]

    got = fn(*T1, *T2)
    if not isinstance(got, tuple):
        got = (got,)
    expected = fwd(*T1, *T2)
    if not isinstance(expected, (list, tuple)):
        expected = (expected,)
    assert len(got) == 2 * l_out + 1
    for i, (a, b) in enumerate(zip(got, expected)):
        b_t = b if torch.is_tensor(b) else torch.full((n,), float(b), dtype=DTYPE)
        a_t = a if torch.is_tensor(a) else torch.full((n,), float(a), dtype=DTYPE)
        diff = (a_t - b_t).abs().max().item()
        assert diff < 1e-11, f"emitter contract {path} component {i}: diff {diff:.3e}"


@pytest.mark.parametrize("path", _ALL_PATHS_L4, ids=lambda p: f"l{p[0]}_l{p[1]}_to_l{p[2]}")
def test_emitted_accum_grad_matches_sympy(path):
    l1, l2, l_out = path
    ns = _exec_generated_module(4)
    fn = ns[f"accum_grad_l{l1}_l{l2}_to_l{l_out}"]
    bwd = compile_even_tp_backward(l1, l2, l_out)

    torch.manual_seed(29 + hash(path) % 1000)
    n = 5
    n1, n2, n_out = 2 * l1 + 1, 2 * l2 + 1, 2 * l_out + 1
    T1 = [torch.randn(n, dtype=DTYPE) for _ in range(n1)]
    T2 = [torch.randn(n, dtype=DTYPE) for _ in range(n2)]
    grad_out = [torch.randn(n, dtype=DTYPE) for _ in range(n_out)]
    init_g1 = [torch.randn(n, dtype=DTYPE) for _ in range(n1)]
    init_g2 = [torch.randn(n, dtype=DTYPE) for _ in range(n2)]

    args = list(grad_out) + list(T1) + list(T2) + list(init_g1) + list(init_g2)
    out = fn(*args)
    if not isinstance(out, tuple):
        out = (out,)
    g1_got = list(out[:n1])
    g2_got = list(out[n1:])

    g1_sympy_plus_g2_sympy = bwd(*T1, *T2, *grad_out)
    g1_sympy = list(g1_sympy_plus_g2_sympy[:n1])
    g2_sympy = list(g1_sympy_plus_g2_sympy[n1:])
    g1_sympy = [s if torch.is_tensor(s) else torch.full((n,), float(s), dtype=DTYPE) for s in g1_sympy]
    g2_sympy = [s if torch.is_tensor(s) else torch.full((n,), float(s), dtype=DTYPE) for s in g2_sympy]

    for i in range(n1):
        expected = init_g1[i] + g1_sympy[i]
        diff = (g1_got[i] - expected).abs().max().item()
        assert diff < 1e-10, f"emitter accum_grad {path} grad_T1[{i}] diff {diff:.3e}"
    for i in range(n2):
        expected = init_g2[i] + g2_sympy[i]
        diff = (g2_got[i] - expected).abs().max().item()
        assert diff < 1e-10, f"emitter accum_grad {path} grad_T2[{i}] diff {diff:.3e}"


def test_emitted_module_matches_expected_api():
    for kernel_l_max in (0, 1, 2, 3, 4):
        ns = _exec_generated_module(kernel_l_max)
        assert ns["KERNEL_L_MAX"] == kernel_l_max
        for path in _tp_path_order(kernel_l_max):
            assert f"contract_l{path[0]}_l{path[1]}_to_l{path[2]}" in ns
            assert f"accum_grad_l{path[0]}_l{path[1]}_to_l{path[2]}" in ns
