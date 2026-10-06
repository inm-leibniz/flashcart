import os

import pytest
import torch

from helpers import random_rotation, requires_cuda
from flashcart import o3
import flashcart.o3._tensor_product as _tp
from flashcart.o3.utils import count_output_paths, rotate_irreps

pytestmark = pytest.mark.usefixtures("fp64_default")


@pytest.fixture(scope="session", autouse=True)
def _pin_tp_autotune_configs():
    if os.environ.get("FLASHCART_TP_TEST_FULL_TUNE") == "1" or not torch.cuda.is_available():
        yield
        return
    from flashcart.o3 import _tensor_product_kernels as _k

    kernels = [
        _k.tp_fwd_kernel,
        _k.tp_bwd_kernel,
        _k.tp_dbwd_kernel,
        _k.tp_fwd_csr_kernel,
        _k.tp_bwd_csr_kernel,
        _k.tp_dbwd_csr_kernel,
    ]
    saved = [(kn, kn.configs, kn.cache) for kn in kernels]
    for kn in kernels:
        kn.configs = kn.configs[:1]
        kn.cache = {}
    yield
    for kn, cfgs, cache in saved:
        kn.configs = cfgs
        kn.cache = cache


def _case(name, l1, l2, lo, f1, f2, sym=False, shared=False, reduce=False, hessian=False):
    return dict(
        name=name,
        l1=l1,
        l2=l2,
        lo=lo,
        f1=f1,
        f2=f2,
        sym=sym,
        shared=shared,
        reduce=reduce,
        hessian=hessian,
    )


_TP_CASES = [
    _case("l0x0to3_f22", 0, 0, 3, 2, 2, hessian=True),
    _case("l3x3to3_f22", 3, 3, 3, 2, 2, hessian=True),
    _case("l3x3to3_f12", 3, 3, 3, 1, 2),
    _case("l3x3to3_f21", 3, 3, 3, 2, 1),
    _case("l3x3to3_f22_shared", 3, 3, 3, 2, 2, shared=True),
    _case("l3x3to3_f22_reduce", 3, 3, 3, 2, 2, reduce=True),
    _case("l3x3to3_f22_shared_reduce", 3, 3, 3, 2, 2, shared=True, reduce=True),
    _case("l3x3to3_f22_sym", 3, 3, 3, 2, 2, sym=True),
    _case("l3x3to0_f22_sym_shared", 3, 3, 0, 2, 2, sym=True, shared=True),
    _case("l3x3to0_f22_shared", 3, 3, 0, 2, 2, shared=True),
    _case("l3x3to2_f22_shared", 3, 3, 2, 2, 2, shared=True),
    _case("l0x3to3_f21", 0, 3, 3, 2, 1),
    _case("l1x3to3_f21", 1, 3, 3, 2, 1),
    _case("l2x3to3_f21", 2, 3, 3, 2, 1),
    _case("l4x4to4_f22", 4, 4, 4, 2, 2, hessian=True),
    _case("l4x4to4_f21", 4, 4, 4, 2, 1),
    _case("l4x3to2_f22", 4, 3, 2, 2, 2),
    _case("l3x3to4_f22_reduce", 3, 3, 4, 2, 2, reduce=True),
]

_L4_OPT_IN = pytest.mark.skipif(
    _tp.KERNEL_L_MAX < 4,
    reason="l=4 kernels compile for minutes; opt in with FLASHCART_KERNEL_L_MAX=4",
)


def _case_params(cases):
    return [
        pytest.param(
            c,
            id=c["name"],
            marks=(_L4_OPT_IN,) if max(c["l1"], c["l2"], c["lo"]) >= 4 else (),
        )
        for c in cases
    ]


_TP_PARAMS = _case_params(_TP_CASES)
_HESSIAN_PARAMS = _case_params([c for c in _TP_CASES if c["hessian"]])

_N_ATOMS = 4
_N_EDGES = 8
_N_NEIGHBORS = 2


def _build_tp(case, use_triton):
    return o3.TensorProduct(
        in1_l_max=case["l1"],
        in2_l_max=case["l2"],
        out_l_max=case["lo"],
        in1_features=case["f1"],
        in2_features=case["f2"],
        symmetric_product=case["sym"],
        shared_weights=case["shared"],
        reduce_paths=case["reduce"],
        use_triton=use_triton,
    )


def _irreps_dim(l_max, features):
    return sum((2 * l + 1) * features for l in range(l_max + 1))


def _make_neighborlist(n_edges, n_neighbors, device=None):
    n_atoms = max(1, (n_edges + n_neighbors - 1) // n_neighbors)
    dst = torch.arange(n_atoms, device=device).repeat_interleave(n_neighbors)[:n_edges]
    src = torch.randint(0, n_atoms - 1, (n_edges,), device=device)
    src = src + (src >= dst).to(src.dtype)
    return n_atoms, src, dst


def _make_inputs(case, tp, device, seed, scatter):
    torch.manual_seed(seed)
    out_features = max(case["f1"], case["f2"])
    if scatter:
        n_atoms, idx_j, idx_i = _make_neighborlist(_N_EDGES, _N_NEIGHBORS, device=device)
        in1 = torch.randn(n_atoms, _irreps_dim(case["l1"], case["f1"]), device=device)
        in2 = torch.randn(_N_EDGES, _irreps_dim(case["l2"], case["f2"]), device=device)
        w_rows = _N_EDGES
    else:
        idx_i = idx_j = None
        in1 = torch.randn(_N_ATOMS, _irreps_dim(case["l1"], case["f1"]), device=device)
        in2 = in1 if case["sym"] else torch.randn(_N_ATOMS, _irreps_dim(case["l2"], case["f2"]), device=device)
        w_rows = _N_ATOMS
    weights = (
        torch.randn(tp.n_total_paths * out_features, device=device)
        if case["shared"]
        else torch.randn(w_rows, tp.n_total_paths * out_features, device=device)
    )
    return in1, in2, weights, idx_i, idx_j


def _leaf_clones(*tensors):
    return [t.detach().clone().requires_grad_(True) for t in tensors]


@pytest.mark.parametrize("scatter", [False, True], ids=["dense", "scatter"])
@pytest.mark.parametrize("case", _TP_PARAMS)
def test_gradcheck_py(case, scatter):
    if scatter and case["sym"]:
        pytest.skip("Scatter does not support symmetric_product.")
    tp = _build_tp(case, use_triton=False)
    in1, in2, weights, idx_i, idx_j = _make_inputs(case, tp, "cpu", seed=0, scatter=scatter)
    in1, in2, weights = _leaf_clones(in1, in2, weights)
    if case["sym"]:
        def fn(a, w):
            return tp(a, a, w, idx_i=idx_i, idx_j=idx_j)

        args = (in1, weights)
    else:

        def fn(a, b, w):
            return tp(a, b, w, idx_i=idx_i, idx_j=idx_j)

        args = (in1, in2, weights)

    assert torch.autograd.gradcheck(fn, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, args, fast_mode=True)


def _parity_setup(case, seed, scatter):
    device = "cuda"
    py_tp = _build_tp(case, use_triton=False).to(device)
    tr_tp = _build_tp(case, use_triton=True).to(device)
    in1, in2, weights, idx_i, idx_j = _make_inputs(case, py_tp, device, seed, scatter)
    return py_tp, tr_tp, in1, in2, weights, idx_i, idx_j


@requires_cuda
@pytest.mark.parametrize("scatter", [False, True], ids=["dense", "scatter"])
@pytest.mark.parametrize("case", _TP_PARAMS)
def test_triton_matches_py_forward(case, scatter):
    if scatter and case["sym"]:
        pytest.skip("Scatter does not support symmetric_product.")
    for seed in [0, 1]:
        py_tp, tr_tp, in1, in2, weights, idx_i, idx_j = _parity_setup(case, seed, scatter)
        with torch.no_grad():
            py_out = py_tp(in1, in2, weights, idx_i=idx_i, idx_j=idx_j)
            tr_out = tr_tp(in1, in2, weights, idx_i=idx_i, idx_j=idx_j)
        assert py_out.shape == tr_out.shape, f"Shape mismatch: {py_out.shape} vs. {tr_out.shape}."
        assert (py_out - tr_out).abs().max() < 1e-12, f"Forward differs: {(py_out - tr_out).abs().max():.6e}."


@requires_cuda
@pytest.mark.parametrize("scatter", [False, True], ids=["dense", "scatter"])
@pytest.mark.parametrize("case", _TP_PARAMS)
def test_triton_matches_py_backward(case, scatter):
    if scatter and case["sym"]:
        pytest.skip("Scatter does not support symmetric_product.")
    py_tp, tr_tp, in1, in2, weights, idx_i, idx_j = _parity_setup(case, 0, scatter)

    grads = {}
    grad_out = None
    for name, tp in [("py", py_tp), ("tr", tr_tp)]:
        a, b, w = _leaf_clones(in1, in2, weights)
        if case["sym"]:
            out = tp(a, a, w, idx_i=idx_i, idx_j=idx_j)
            leaves = [a, w]
        else:
            out = tp(a, b, w, idx_i=idx_i, idx_j=idx_j)
            leaves = [a, b, w]
        if grad_out is None:
            torch.manual_seed(99)
            grad_out = torch.randn_like(out)
        grads[name] = torch.autograd.grad(out, leaves, grad_outputs=grad_out)

    for g_py, g_tr in zip(grads["py"], grads["tr"]):
        assert (g_py - g_tr).abs().max() < 1e-12, f"Gradient differs: {(g_py - g_tr).abs().max():.6e}."


@requires_cuda
@pytest.mark.parametrize("scatter", [False, True], ids=["dense", "scatter"])
@pytest.mark.parametrize("case", _TP_PARAMS)
def test_triton_matches_py_double_backward(case, scatter):
    if scatter and case["sym"]:
        pytest.skip("Scatter does not support symmetric_product.")
    py_tp, tr_tp, in1, in2, weights, idx_i, idx_j = _parity_setup(case, 0, scatter)

    second = {}
    v_base = None
    for name, tp in [("py", py_tp), ("tr", tr_tp)]:
        a, b, w = _leaf_clones(in1, in2, weights)
        if case["sym"]:
            leaves = [a, w]
            out = tp(a, a, w, idx_i=idx_i, idx_j=idx_j)
        else:
            leaves = [a, b, w]
            out = tp(a, b, w, idx_i=idx_i, idx_j=idx_j)
        if v_base is None:
            torch.manual_seed(199)
            v_base = torch.randn_like(out)
        v = v_base.detach().clone().requires_grad_(True)
        first = torch.autograd.grad((out * v).sum(), leaves, create_graph=True)
        second[name] = torch.autograd.grad(sum(g.sum() for g in first), leaves + [v])

    for d_py, d_tr in zip(second["py"], second["tr"]):
        assert (d_py - d_tr).abs().max() < 1e-10, f"Second-order gradient differs: {(d_py - d_tr).abs().max():.6e}."


def _hessian_block(tp, in1, in2, w, v, wrt_a, wrt_b, rows, idx_i=None, idx_j=None):
    device, dt = in1.device, in1.dtype
    kwargs = {} if idx_i is None else {"idx_i": idx_i, "idx_j": idx_j}
    in1_l, in2_l, w_l, v_l = _leaf_clones(in1, in2, w, v)
    leaves = {"in1": in1_l, "in2": in2_l, "w": w_l, "v": v_l}
    n_b = leaves[wrt_b].numel()
    g_a = torch.autograd.grad(
        (tp(in1_l, in2_l, w_l, **kwargs) * v_l).sum(),
        leaves[wrt_a],
        create_graph=True,
    )[
        0
    ].reshape(-1)
    out_rows = []
    for i in rows:
        if g_a[i].requires_grad:
            row = torch.autograd.grad(g_a[i], leaves[wrt_b], retain_graph=True, allow_unused=True)[0]
            row = row.reshape(-1) if row is not None else torch.zeros(n_b, device=device, dtype=dt)
        else:
            row = torch.zeros(n_b, device=device, dtype=dt)
        out_rows.append(row)
    return torch.stack(out_rows)


_HESSIAN_PAIRS = [
    ("in1", "in1"),
    ("in2", "in2"),
    ("w", "w"),
    ("v", "v"),
    ("in1", "in2"),
    ("in1", "w"),
    ("in1", "v"),
    ("in2", "w"),
    ("in2", "v"),
    ("w", "v"),
]


_HESSIAN_ROWS = int(os.environ.get("FLASHCART_TP_HESSIAN_ROWS", "6"))


@requires_cuda
@pytest.mark.parametrize("scatter", [False, True], ids=["dense", "scatter"])
@pytest.mark.parametrize("case", _HESSIAN_PARAMS)
def test_triton_matches_py_hessian(case, scatter):
    if case["sym"]:
        pytest.skip("Hessian parity uses independent in1/in2 leaves.")
    if scatter and case["sym"]:
        pytest.skip("Scatter does not support symmetric_product.")
    py_tp, tr_tp, in1, in2, weights, idx_i, idx_j = _parity_setup(case, 0, scatter)

    torch.manual_seed(99)
    out_tmp = py_tp(in1, in2, weights, idx_i=idx_i, idx_j=idx_j)
    v = torch.randn_like(out_tmp)

    for pair_idx, (wrt_a, wrt_b) in enumerate(_HESSIAN_PAIRS):
        n_a = {"in1": in1, "in2": in2, "w": weights, "v": v}[wrt_a].numel()
        if _HESSIAN_ROWS > 0 and n_a > _HESSIAN_ROWS:
            gen = torch.Generator().manual_seed(1000 + pair_idx)
            rows = torch.randperm(n_a, generator=gen)[:_HESSIAN_ROWS].tolist()
        else:
            rows = list(range(n_a))
        H_py = _hessian_block(py_tp, in1, in2, weights, v, wrt_a, wrt_b, rows, idx_i, idx_j)
        H_tr = _hessian_block(tr_tp, in1, in2, weights, v, wrt_a, wrt_b, rows, idx_i, idx_j)
        assert (
            H_py - H_tr
        ).abs().max() < 1e-10, f"Hessian d2f/d({wrt_a})d({wrt_b}) differs: {(H_py - H_tr).abs().max():.6e}."


_EQUIV_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("scatter", [False, True], ids=["dense", "scatter"])
@pytest.mark.parametrize("device", _EQUIV_DEVICES)
@pytest.mark.parametrize("case", _TP_PARAMS)
def test_rotation_equivariance(case, device, scatter):
    if scatter and case["sym"]:
        pytest.skip("Scatter does not support symmetric_product.")
    use_triton = device == "cuda"
    tp = _build_tp(case, use_triton=use_triton).to(device)

    for seed in [0, 1]:
        torch.manual_seed(seed)
        in1, in2, weights, idx_i, idx_j = _make_inputs(case, tp, device, seed, scatter)
        R = random_rotation(seed=seed).to(device)

        in1_rot = rotate_irreps(in1, R, case["l1"])
        in2_rot = in1_rot if (case["sym"] and not scatter) else rotate_irreps(in2, R, case["l2"])
        out = tp(in1, in2, weights, idx_i=idx_i, idx_j=idx_j)
        out_rot = tp(in1_rot, in2_rot, weights, idx_i=idx_i, idx_j=idx_j)
        n_paths_rot = [min(1, x) for x in tp.out_paths] if case["reduce"] else tp.out_paths
        out_then_rot = rotate_irreps(out, R, case["lo"], n_paths=n_paths_rot)

        assert torch.allclose(
            out_rot, out_then_rot, atol=1e-10
        ), f"Rotation equivariance failed. Max diff: {(out_rot - out_then_rot).abs().max():.6e}"


def _make_neighborlist_int32(n_edges, n_neighbors, device):
    n_atoms = max(2, n_edges // n_neighbors)
    dst = torch.arange(n_atoms, device=device).repeat_interleave(n_neighbors)[:n_edges]
    src = torch.randint(0, n_atoms - 1, (n_edges,), device=device)
    src = src + (src >= dst).to(src.dtype)
    return n_atoms, src.to(torch.int32), dst.to(torch.int32)


def _build_internals_tp(l_max, f1, f2, shared_weights, reduce_paths, use_triton, device):
    return o3.TensorProduct(
        in1_l_max=l_max,
        in2_l_max=l_max,
        out_l_max=l_max,
        in1_features=f1,
        in2_features=f2,
        symmetric_product=False,
        shared_weights=shared_weights,
        reduce_paths=reduce_paths,
        use_triton=use_triton,
    ).to(device)


def test_csr_path_groups_partition():
    n_paths_per_l = [3, 4, 4, 4]
    groups = _tp._csr_path_groups(3, n_paths_per_l, reduce_paths=False)
    assert groups == sorted(groups), "groups must be contiguous rank ranges"
    assert groups[0] == 0
    comps = [(2 * l + 1) * n for l, n in enumerate(n_paths_per_l)]
    effective = max(_tp.TUNING.csr_group_budget, max(comps))
    for g in range(max(groups) + 1):
        used = sum(c for c, gr in zip(comps, groups) if gr == g)
        assert used <= effective


@requires_cuda
@pytest.mark.parametrize("budget", [7, 30])
@pytest.mark.parametrize("f1,f2", [(4, 1), (4, 4)])
@pytest.mark.parametrize("shared_weights,reduce_paths", [(False, False), (True, False), (False, True)])
def test_grouped_scatter_matches_reference(monkeypatch, budget, f1, f2, shared_weights, reduce_paths):
    monkeypatch.setattr(_tp.TUNING, "csr_group_budget", budget)
    device = "cuda"
    l_max = 3
    for seed in [0, 1]:
        torch.manual_seed(seed)
        n_atoms, idx_j, idx_i = _make_neighborlist_int32(64, 4, device)
        in1_dim = sum((2 * l + 1) * f1 for l in range(l_max + 1))
        in2_dim = sum((2 * l + 1) * f2 for l in range(l_max + 1))
        in1 = torch.randn(n_atoms, in1_dim, device=device)
        in2 = torch.randn(64, in2_dim, device=device)

        reference = _build_internals_tp(l_max, f1, f2, shared_weights, reduce_paths, False, device)
        impl = _build_internals_tp(l_max, f1, f2, shared_weights, reduce_paths, True, device)
        n_total_paths = reference.n_total_paths
        out_features = max(f1, f2)
        w = (
            torch.randn(n_total_paths * out_features, device=device)
            if shared_weights
            else torch.randn(64, n_total_paths * out_features, device=device)
        )

        for _ in range(2):
            args_ref = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
            args_impl = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
            out_ref = reference(*args_ref, idx_i=idx_i, idx_j=idx_j)
            out_impl = impl(*args_impl, idx_i=idx_i, idx_j=idx_j)
            grad_out = torch.randn_like(out_ref)
            out_ref.backward(grad_out)
            out_impl.backward(grad_out)

            assert (out_ref - out_impl).abs().max() < 1e-12
            for a, b in zip(args_ref, args_impl):
                assert (a.grad - b.grad).abs().max() < 1e-12


def test_bwd_csr_chunk_slots_match_expected_table():
    assert _tp._bwd_csr_chunk_slots(32 * 1024, 1024) == 1
    assert _tp._bwd_csr_chunk_slots(64 * 1024, 1024) == 2
    assert _tp._bwd_csr_chunk_slots(128 * 1024, 1024) == 4
    assert _tp._bwd_csr_chunk_slots(4096 * 64, 64) == 64
    assert _tp._bwd_csr_chunk_slots(0, 0) == 1


def _pin_chunked_config(monkeypatch, chunk):
    import triton

    from flashcart.o3 import _tensor_product_kernels as _k

    monkeypatch.setattr(
        _k.tp_bwd_csr_kernel,
        "configs",
        [triton.Config({"FEATURE_BLOCK": 16, "CHUNK": chunk}, num_warps=1, num_stages=1)],
    )
    monkeypatch.setattr(_k.tp_bwd_csr_kernel, "cache", {})


@requires_cuda
@pytest.mark.parametrize("budget", [None, 30])
@pytest.mark.parametrize("f1,f2", [(4, 1), (2, 2)])
@pytest.mark.parametrize("shared_weights,reduce_paths", [(False, False), (True, False), (False, True)])
def test_chunked_bwd_matches_reference(monkeypatch, budget, f1, f2, shared_weights, reduce_paths):
    if budget is not None:
        monkeypatch.setattr(_tp.TUNING, "bwd_group_budget", budget)
    monkeypatch.setattr(_tp.TUNING, "bwd_csr_force_slots", 2)
    _pin_chunked_config(monkeypatch, chunk=2)
    device = "cuda"
    l_max = 3
    for seed in [0, 1]:
        torch.manual_seed(seed)
        n_atoms, idx_j, idx_i = _make_neighborlist_int32(64, 4, device)
        in1_dim = sum((2 * l + 1) * f1 for l in range(l_max + 1))
        in2_dim = sum((2 * l + 1) * f2 for l in range(l_max + 1))
        in1 = torch.randn(n_atoms, in1_dim, device=device)
        in2 = torch.randn(64, in2_dim, device=device)

        reference = _build_internals_tp(l_max, f1, f2, shared_weights, reduce_paths, False, device)
        impl = _build_internals_tp(l_max, f1, f2, shared_weights, reduce_paths, True, device)
        n_total_paths = reference.n_total_paths
        out_features = max(f1, f2)
        w = (
            torch.randn(n_total_paths * out_features, device=device)
            if shared_weights
            else torch.randn(64, n_total_paths * out_features, device=device)
        )

        for _ in range(2):
            args_ref = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
            args_impl = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
            out_ref = reference(*args_ref, idx_i=idx_i, idx_j=idx_j)
            out_impl = impl(*args_impl, idx_i=idx_i, idx_j=idx_j)
            grad_out = torch.randn_like(out_ref)
            out_ref.backward(grad_out)
            out_impl.backward(grad_out)

            assert (out_ref - out_impl).abs().max() < 1e-12
            for a, b in zip(args_ref, args_impl):
                scale = b.grad.abs().max().clamp(min=1.0)
                assert (a.grad - b.grad).abs().max() / scale < 1e-13


@requires_cuda
def test_chunked_bwd_uneven_degrees(monkeypatch):
    monkeypatch.setattr(_tp.TUNING, "bwd_csr_force_slots", 3)
    _pin_chunked_config(monkeypatch, chunk=2)
    device = "cuda"
    l_max, f1, f2 = 3, 4, 1
    torch.manual_seed(0)
    degrees = [0, 1, 5, 26]
    n_atoms = len(degrees)
    idx_i = torch.repeat_interleave(torch.arange(n_atoms, device=device), torch.tensor(degrees, device=device)).to(
        torch.int32
    )
    n_edges = int(sum(degrees))
    idx_j = torch.randint(0, n_atoms, (n_edges,), device=device, dtype=torch.int32)
    in1_dim = sum((2 * l + 1) * f1 for l in range(l_max + 1))
    in2_dim = sum((2 * l + 1) * f2 for l in range(l_max + 1))
    in1 = torch.randn(n_atoms, in1_dim, device=device)
    in2 = torch.randn(n_edges, in2_dim, device=device)

    reference = _build_internals_tp(l_max, f1, f2, False, False, False, device)
    impl = _build_internals_tp(l_max, f1, f2, False, False, True, device)
    w = torch.randn(n_edges, reference.n_total_paths * f1, device=device)

    for _ in range(2):
        args_ref = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
        args_impl = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
        out_ref = reference(*args_ref, idx_i=idx_i, idx_j=idx_j)
        out_impl = impl(*args_impl, idx_i=idx_i, idx_j=idx_j)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_impl.backward(grad_out)
        assert (out_ref - out_impl).abs().max() < 1e-12
        for a, b in zip(args_ref, args_impl):
            scale = b.grad.abs().max().clamp(min=1.0)
            assert (a.grad - b.grad).abs().max() / scale < 1e-13


@requires_cuda
@pytest.mark.parametrize("f1,f2", [(4, 1), (4, 4)])
def test_edge_scatter_fallback_matches_reference(monkeypatch, f1, f2):
    monkeypatch.setattr(_tp.TUNING, "scatter_kernel", "edge")
    device = "cuda"
    l_max = 3
    torch.manual_seed(0)
    n_atoms, idx_j, idx_i = _make_neighborlist_int32(64, 4, device)
    in1_dim = sum((2 * l + 1) * f1 for l in range(l_max + 1))
    in2_dim = sum((2 * l + 1) * f2 for l in range(l_max + 1))
    in1 = torch.randn(n_atoms, in1_dim, device=device)
    in2 = torch.randn(64, in2_dim, device=device)

    reference = _build_internals_tp(l_max, f1, f2, False, False, False, device)
    impl = _build_internals_tp(l_max, f1, f2, False, False, True, device)
    w = torch.randn(64, reference.n_total_paths * max(f1, f2), device=device)

    for _ in range(2):
        args_ref = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
        args_impl = [t.detach().clone().requires_grad_(True) for t in (in1, in2, w)]
        out_ref = reference(*args_ref, idx_i=idx_i, idx_j=idx_j)
        out_impl = impl(*args_impl, idx_i=idx_i, idx_j=idx_j)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_impl.backward(grad_out)
        assert (out_ref - out_impl).abs().max() < 1e-12
        for a, b in zip(args_ref, args_impl):
            assert (a.grad - b.grad).abs().max() < 1e-12

    ref = _second_order(reference, in1, in2, w, idx_i, idx_j, mode="train")
    res = _second_order(impl, in1, in2, w, idx_i, idx_j, mode="train")
    for a, b in zip(ref, res):
        scale = b.abs().max().clamp(min=1.0)
        assert (a - b).abs().max() / scale < 1e-13


def _second_order(tp, in1, in2, w, idx_i, idx_j, mode):
    i1 = in1.detach().clone().requires_grad_(True)
    i2 = in2.detach().clone().requires_grad_(True)
    wi = w.detach().clone().requires_grad_(True)
    out = tp(i1, i2, wi, idx_i=idx_i, idx_j=idx_j)
    torch.manual_seed(7)
    if mode == "ones":
        grads = torch.autograd.grad(out.sum(), (i1, i2), create_graph=True)
        d = torch.autograd.grad(sum(g.square().sum() for g in grads), (i1, i2, wi))
        return [g.detach() for g in grads] + [t.detach() for t in d]
    c = torch.randn_like(out).requires_grad_(True)
    first_wrt = (i1, i2) if mode == "no_vw" else (i1, i2, wi)
    grads = torch.autograd.grad((out * c).sum(), first_wrt, create_graph=True)
    mults = [torch.randn_like(g) for g in grads]
    loss = sum((g * m).sum() for g, m in zip(grads, mults))
    wrt = (c,) if mode == "a_only" else (i1, i2, wi, c)
    d = torch.autograd.grad(loss, wrt)
    return [g.detach() for g in grads] + [t.detach() for t in d]


def _run_csr_dbwd_case(monkeypatch, plan, budget, f1, f2, shared_weights, reduce_paths, mode):
    if plan is not None:
        monkeypatch.setattr(_tp.TUNING, "dbwd_split_override", plan)
    if budget is not None:
        monkeypatch.setattr(_tp.TUNING, "csr_group_budget", budget)
        monkeypatch.setattr(_tp.TUNING, "bwd_group_budget", budget)
    device = "cuda"
    l_max = 3
    torch.manual_seed(0)
    n_atoms, idx_j, idx_i = _make_neighborlist_int32(64, 4, device)
    in1_dim = sum((2 * l + 1) * f1 for l in range(l_max + 1))
    in2_dim = sum((2 * l + 1) * f2 for l in range(l_max + 1))
    in1 = torch.randn(n_atoms, in1_dim, device=device)
    in2 = torch.randn(64, in2_dim, device=device)

    reference = _build_internals_tp(l_max, f1, f2, shared_weights, reduce_paths, False, device)
    impl = _build_internals_tp(l_max, f1, f2, shared_weights, reduce_paths, True, device)
    out_features = max(f1, f2)
    n_total_paths = reference.n_total_paths
    w = (
        torch.randn(n_total_paths * out_features, device=device)
        if shared_weights
        else torch.randn(64, n_total_paths * out_features, device=device)
    )

    for _ in range(2):
        ref = _second_order(reference, in1, in2, w, idx_i, idx_j, mode)
        res = _second_order(impl, in1, in2, w, idx_i, idx_j, mode)
        for a, b in zip(ref, res):
            scale = b.abs().max().clamp(min=1.0)
            assert (a - b).abs().max() / scale < 1e-13


@requires_cuda
@pytest.mark.parametrize(
    ("plan", "budget", "f1", "f2", "shared_weights", "reduce_paths", "mode"),
    [
        pytest.param("fused", None, 8, 1, False, False, "train", id="fused"),
        pytest.param("fused", 30, 8, 1, False, False, "train", id="fused-grouped"),
        pytest.param("full", None, 8, 1, False, False, "train", id="full"),
        pytest.param("full", 30, 8, 1, False, False, "train", id="full-grouped"),
        pytest.param("full", None, 2, 2, False, False, "train", id="in2_features"),
        pytest.param("full", 30, 2, 2, False, False, "train", id="in2_features-grouped"),
        pytest.param("full", None, 8, 1, True, False, "train", id="shared_weights"),
        pytest.param("full", 30, 8, 1, True, False, "train", id="shared_weights-grouped"),
        pytest.param("full", None, 8, 1, False, True, "train", id="reduce_paths"),
        pytest.param("full", 30, 8, 1, False, True, "train", id="reduce_paths-grouped"),
        pytest.param("full", 30, 8, 1, False, False, "ones", id="need_masks-ones"),
        pytest.param("full", 30, 8, 1, False, False, "a_only", id="need_masks-a_only"),
        pytest.param("full", 30, 8, 1, False, False, "no_vw", id="need_masks-no_vw"),
    ],
)
def test_csr_double_backward_matches_reference(monkeypatch, plan, budget, f1, f2, shared_weights, reduce_paths, mode):
    _run_csr_dbwd_case(monkeypatch, plan, budget, f1, f2, shared_weights, reduce_paths, mode)


@requires_cuda
def test_grouped_scatter_double_backward(monkeypatch):
    monkeypatch.setattr(_tp.TUNING, "csr_group_budget", 30)
    device = "cuda"
    l_max, f1, f2 = 3, 4, 1
    torch.manual_seed(0)
    n_atoms, idx_j, idx_i = _make_neighborlist_int32(64, 4, device)
    in1_dim = sum((2 * l + 1) * f1 for l in range(l_max + 1))
    in2_dim = sum((2 * l + 1) * f2 for l in range(l_max + 1))
    in1 = torch.randn(n_atoms, in1_dim, device=device)
    in2 = torch.randn(64, in2_dim, device=device)

    reference = _build_internals_tp(l_max, f1, f2, False, False, False, device)
    impl = _build_internals_tp(l_max, f1, f2, False, False, True, device)
    w = torch.randn(64, reference.n_total_paths * f1, device=device)

    results = {}
    for name, tp in (("ref", reference), ("impl", impl)):
        i1 = in1.detach().clone().requires_grad_(True)
        i2 = in2.detach().clone().requires_grad_(True)
        wi = w.detach().clone().requires_grad_(True)
        out = tp(i1, i2, wi, idx_i=idx_i, idx_j=idx_j)
        grads = torch.autograd.grad(out.sum(), (i1, i2), create_graph=True)
        sum(g.square().sum() for g in grads).backward()
        results[name] = (grads[0].detach(), grads[1].detach(), i1.grad, i2.grad, wi.grad)

    for a, b in zip(results["ref"], results["impl"]):
        scale = b.abs().max().clamp(min=1.0)
        assert (a - b).abs().max() / scale < 1e-13
