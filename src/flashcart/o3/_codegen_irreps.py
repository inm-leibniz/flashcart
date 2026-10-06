"""Generate irreducible Cartesian expansions and their derivatives symbolically.

The expansion implements Eq. (1) of Zaverkin et al., NeurIPS 2024. For a
unit direction vector, the rank-l symmetric traceless tensor is::

    T^l(r_hat) = (2l - 1)!! / l!
        * sum_{m=0}^{floor(l/2)} (-1)^m
        * (2l - 2m - 1)!! / (2l - 1)!!
        * {r_hat^{otimes (l - 2m)} otimes I^{otimes m}}

The braces denote the sum over distinct placements of vector indices and
Kronecker-delta pairs. There are ``l! / ((l - 2m)! * m! * 2**m)`` placements,
and the sum is not divided by their number. Only the ``2*l + 1`` independent
components are emitted, in the order defined by ``stored_basis``.

For component expressions ``T_a(x)``, symbolic differentiation gives the
input gradient ``grad_x_k = sum_a g_a * dT_a/dx_k``. Differentiating its
contraction with a vector ``v`` gives ``d_grad_g_a = sum_k v_k * dT_a/dx_k``
and ``d_grad_x_k = d/dx_k (v dot grad_x)``. The numerical reference callables
and generated kernels use these same expressions.

The PyTorch generator emits forward expressions and uses automatic
differentiation. The Triton generator emits forward, backward, and
double-backward kernels. Each program handles a block of input rows, and
compile-time rank guards remove inactive expressions. The autotuning key
uses ``L_MAX`` to reuse the selected launch configuration for that maximum
rank. Triton compiles the required kernel specializations separately.

The Python module ``flashcart.o3._irreps`` loads the generated source from the
cache and registers its derivative operations with PyTorch. The rank-zero
component is one and contributes no input derivatives.
"""

import math
from functools import lru_cache, partial
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import sympy as sp

from flashcart.o3._codegen_common import (
    _lambdify,
    _pycode,
    double_factorial,
    stored_basis,
)


def _pair_partitions(positions: Tuple[int, ...]):
    if not positions:
        yield []
        return
    first = positions[0]
    rest = positions[1:]
    for i in range(len(rest)):
        partner = rest[i]
        remaining = rest[:i] + rest[i + 1 :]
        for sub in _pair_partitions(remaining):
            yield [(first, partner), *sub]


def _sym_distinct(
    l: int,
    indices: Tuple[int, ...],
    vec_tags_counts: Dict[str, int],
    num_I_pairs: int,
    vec_values: Dict[str, Sequence[sp.Expr]],
) -> sp.Expr:
    """Sum distinct placements of vector slots and Kronecker-delta pairs.

    Each distinct placement contributes once. The result is not divided by the
    number of placements, unlike a normalized symmetrization average.

    Args:
        l (int): Total number of Cartesian index slots.
        indices (tuple[int, ...]): Cartesian indices for the requested component.
        vec_tags_counts (dict[str, int]): Number of slots occupied by each vector.
        num_I_pairs (int): Number of Kronecker-delta pairs.
        vec_values (dict[str, Sequence[sympy.Expr]]): Three components for each
            vector tag.

    Returns:
        sympy.Expr: Distinct-placement sum for the requested component.

    Raises:
        ValueError: Vector and delta slots do not total ``l``, or the component
            does not contain ``l`` Cartesian indices.
    """
    total_vec = sum(vec_tags_counts.values())
    if total_vec + 2 * num_I_pairs != l:
        raise ValueError("Slot counts inconsistent with l: " f"vector slots={total_vec}, I-pairs={num_I_pairs}, l={l}.")
    if len(indices) != l:
        raise ValueError(f"Expected {l} indices, got {len(indices)}.")

    distinct_tags = list(vec_tags_counts.keys())
    result = sp.Integer(0)

    def recurse(remaining: Tuple[int, ...], tags_idx: int):
        if tags_idx == len(distinct_tags):
            if not remaining:
                yield {}, []
                return
            for partition in _pair_partitions(remaining):
                yield {}, partition
            return
        tag = distinct_tags[tags_idx]
        count = vec_tags_counts[tag]
        for selected in combinations(remaining, count):
            sel_set = set(selected)
            rest = tuple(p for p in remaining if p not in sel_set)
            for sub_map, partition in recurse(rest, tags_idx + 1):
                merged = {tag: selected, **sub_map}
                yield merged, partition

    for tag_positions, partition in recurse(tuple(range(l)), 0):
        skip = False
        for p1, p2 in partition:
            if indices[p1] != indices[p2]:
                skip = True
                break
        if skip:
            continue
        contrib: sp.Expr = sp.Integer(1)
        for tag, positions in tag_positions.items():
            vec = vec_values[tag]
            for pos in positions:
                contrib *= vec[indices[pos]]
        result += contrib

    return result


_X = sp.Symbol("x", real=True)
_Y = sp.Symbol("y", real=True)
_Z = sp.Symbol("z", real=True)
_VX = sp.Symbol("vx", real=True)
_VY = sp.Symbol("vy", real=True)
_VZ = sp.Symbol("vz", real=True)


@lru_cache(maxsize=None)
def _g_symbols(l: int) -> Tuple[sp.Symbol, ...]:
    n = 2 * l + 1
    return tuple(sp.symbols(f"g0:{n}", real=True))


@lru_cache(maxsize=None)
def _forward_components(l: int) -> Tuple[sp.Expr, ...]:
    vec_x = [_X, _Y, _Z]
    C = sp.Rational(double_factorial(2 * l - 1), math.factorial(l))
    denom = sp.Integer(double_factorial(2 * l - 1))

    components: List[sp.Expr] = []
    for a, b, c in stored_basis(l):
        indices = tuple([0] * a + [1] * b + [2] * c)
        expr: sp.Expr = sp.Integer(0)
        for m in range(l // 2 + 1):
            p = l - 2 * m
            prefactor = sp.Integer((-1) ** m * double_factorial(2 * l - 2 * m - 1)) / denom
            tags = {"x": p} if p > 0 else {}
            contribution = _sym_distinct(l, indices, tags, m, {"x": vec_x})
            expr += C * prefactor * contribution
        components.append(sp.expand(expr))
    return tuple(components)


def irreps_forward_sympy(l: int) -> List[sp.Expr]:
    """Construct symbolic independent components of a rank-l direction tensor.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        list[sympy.Expr]: The ``2*l + 1`` polynomials in ``x``, ``y``, and ``z``,
            ordered by ``stored_basis(l)``. The vector is assumed to be normalized.
    """
    return list(_forward_components(l))


def irreps_backward_sympy(l: int) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Construct the input-gradient expressions for a Cartesian expansion.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        tuple[sympy.Expr, sympy.Expr, sympy.Expr]: Gradients with respect to
            ``x``, ``y``, and ``z`` after contraction with the incoming
            component gradients ``g0`` to ``g{2*l}``.
    """
    n = 2 * l + 1
    g = _g_symbols(l)
    forward = _forward_components(l)

    grad_x = sum(g[a] * sp.diff(forward[a], _X) for a in range(n))
    grad_y = sum(g[a] * sp.diff(forward[a], _Y) for a in range(n))
    grad_z = sum(g[a] * sp.diff(forward[a], _Z) for a in range(n))

    return sp.expand(grad_x), sp.expand(grad_y), sp.expand(grad_z)


def irreps_double_backward_sympy(
    l: int,
) -> Tuple[List[sp.Expr], Tuple[sp.Expr, sp.Expr, sp.Expr]]:
    """Differentiate the Cartesian expansion input-gradient expressions.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        tuple[list[sympy.Expr], tuple[sympy.Expr, sympy.Expr, sympy.Expr]]:
            Derivatives with respect to the incoming component gradient,
            followed by derivatives with respect to ``x``, ``y``, and ``z``.
            The three input-gradient seeds are ``vx``, ``vy``, and ``vz``.
    """
    n = 2 * l + 1
    forward = _forward_components(l)

    dgrad_g: List[sp.Expr] = []
    for a in range(n):
        expr = _VX * sp.diff(forward[a], _X) + _VY * sp.diff(forward[a], _Y) + _VZ * sp.diff(forward[a], _Z)
        dgrad_g.append(sp.expand(expr))

    grad_x, grad_y, grad_z = irreps_backward_sympy(l)
    v_dot_grad = _VX * grad_x + _VY * grad_y + _VZ * grad_z
    dgrad_x = sp.expand(sp.diff(v_dot_grad, _X))
    dgrad_y = sp.expand(sp.diff(v_dot_grad, _Y))
    dgrad_z = sp.expand(sp.diff(v_dot_grad, _Z))

    return dgrad_g, (dgrad_x, dgrad_y, dgrad_z)


# --------------------------
# Codegen for PyTorch irreps
# --------------------------


@lru_cache(maxsize=None)
def compile_forward(l: int):
    """Build a cached numerical reference callable for one expansion rank.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        Callable: Function accepting ``x``, ``y``, and ``z``. Its result lists
            the stored forward components.
    """
    return _lambdify((_X, _Y, _Z), list(_forward_components(l)))


@lru_cache(maxsize=None)
def compile_backward(l: int):
    """Build a cached numerical reference callable for one expansion rank.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        Callable: Function accepting ``x``, ``y``, and ``z``, followed by the
            stored output-gradient components. Its result lists the three
            input-gradient components.
    """
    args = (_X, _Y, _Z, *_g_symbols(l))
    gx, gy, gz = irreps_backward_sympy(l)
    return _lambdify(args, [gx, gy, gz])


@lru_cache(maxsize=None)
def compile_double_backward(l: int):
    """Build a cached numerical reference callable for one expansion rank.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        Callable: Function accepting ``x``, ``y``, ``z``, the stored output-gradient components,
            then ``vx``, ``vy``, and ``vz``. Its result lists
            the derivatives with respect to the stored output gradient, followed
            by the three input derivatives.
    """
    args = (_X, _Y, _Z, *_g_symbols(l), _VX, _VY, _VZ)
    dgrad_g, (dx, dy, dz) = irreps_double_backward_sympy(l)
    return _lambdify(args, [*dgrad_g, dx, dy, dz])


def build_py_irreps_forward_module_source(kernel_l_max: int) -> str:
    """Generate a PyTorch Cartesian expansion module as Python source.

    Args:
        kernel_l_max (int): Maximum tensor rank included in the generated module.

    Returns:
        str: Python module source. The cache layer adds the fingerprint header
            when writing the module.
    """
    lines: List[str] = [
        '"""Auto-generated by flashcart.o3._codegen_irreps.build_py_irreps_forward_module_source."""',
        "",
        "import torch",
        "",
        f"KERNEL_L_MAX = {kernel_l_max}",
        "",
    ]
    subs = {
        _X: sp.Symbol("x_0", real=True),
        _Y: sp.Symbol("x_1", real=True),
        _Z: sp.Symbol("x_2", real=True),
    }
    lines.extend(
        [
            "def py_irreps_forward(x, l_max: int):",
            "    l_max = int(l_max)",
            "    if l_max > KERNEL_L_MAX:",
            "        raise NotImplementedError(",
            '            f"Generated py_irreps supports l_max <= {KERNEL_L_MAX}, got {l_max}."',
            "        )",
            "    x_0 = x[:, 0]",
            "    x_1 = x[:, 1]",
            "    x_2 = x[:, 2]",
            "    components = []",
            "    components.append(torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device))",
        ]
    )

    for l in range(1, kernel_l_max + 1):
        comp_names: List[str] = []
        lines.append(f"    if l_max >= {l}:")
        for i, expr in enumerate(_forward_components(l)):
            name = f"c_l{l}_{i}"
            comp_names.append(name)
            rhs = _pycode(expr.xreplace(subs))
            lines.append(f"        {name} = {rhs}")
        lines.append(f"        components.append(torch.stack([{', '.join(comp_names)}], dim=-1))")
    lines.extend(
        [
            "    return torch.cat(components, dim=-1)",
            "",
        ]
    )
    return "\n".join(lines)


@lru_cache(maxsize=None)
def compile_py_irreps_forward(l_max: int):
    """Generate a PyTorch expansion for a specified maximum tensor rank.

    This provides the fallback used by ``flashcart.o3._irreps.py_irreps`` above the rank
    limit of the cached module. The returned callable is cached for each ``l_max``.

    Args:
        l_max (int): Maximum tensor rank to include.

    Returns:
        Callable: Function accepting unit vectors of shape ``(n, 3)`` and returning
            their independent components of shape ``(n, (l_max + 1)**2)``.
            The callable does not normalize or check its inputs.
    """
    namespace: dict = {}
    src = build_py_irreps_forward_module_source(l_max)
    exec(src, namespace)  # noqa: S102 — trusted SymPy-emitted Python source
    return partial(namespace["py_irreps_forward"], l_max=l_max)


# -------------------------
# Codegen for Triton irreps
# -------------------------


_MODULE_DOCSTRING = '"""Auto-generated by flashcart.o3._codegen_irreps.build_triton_irreps_module_source."""'

_FWD_CONFIGS_SOURCE = """\
# The kernels are loop-free and elementwise per row (a few loads, up to
# (L_MAX + 1)**2 stores each), so there is nothing for num_stages to
# pipeline. key=["L_MAX"] caches one config per rank per process; tiny
# blocks are deliberately absent so a small first-seen batch cannot lock
# in a config that starves large batches.
_IRREPS_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
]
"""

_BWD_CONFIGS_SOURCE = """\
_IRREPS_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
]
"""

_DBWD_CONFIGS_SOURCE = """\
# One notch smaller than fwd/bwd: the double-backward polynomial bodies are
# the largest emitted, and fp64 doubles the per-lane register cost.
_IRREPS_DBWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
]
"""


def _emit_row_prologue(lines: List[str]) -> None:
    lines.append("    pid = tl.program_id(0)")
    lines.append("    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
    lines.append("    mask = offsets < n_batch")
    lines.append("")


def _emit_vec3_load(lines: List[str], names: Tuple[str, str, str], ptr: str, stride: str) -> None:
    for k, name in enumerate(names):
        lines.append(f"    {name} = tl.load({ptr} + offsets * {stride} + {k}, mask=mask, other=0.0)")
    lines.append("")


def _emit_irreps_fwd_kernel(kernel_l_max: int) -> str:
    lines: List[str] = []
    lines.append('@triton.autotune(configs=_IRREPS_FWD_AUTOTUNE_CONFIGS, key=["L_MAX"])')
    lines.append("@triton.jit")
    lines.append("def irreps_fwd_kernel(")
    lines.append("    x_ptr,")
    lines.append("    out_ptr,")
    lines.append("    n_batch,")
    lines.append("    x_stride_n,")
    lines.append("    out_stride_n,")
    lines.append("    L_MAX: tl.constexpr,")
    lines.append("    BLOCK_SIZE: tl.constexpr,")
    lines.append("):")
    lines.append('    """All stored components of ``T^l(x)`` for ``l <= L_MAX``, one row per lane."""')
    _emit_row_prologue(lines)
    _emit_vec3_load(lines, ("x", "y", "z"), "x_ptr", "x_stride_n")
    lines.append("    # l = 0 (constant 1)")
    lines.append("    tl.store(out_ptr + offsets * out_stride_n + 0, 1.0, mask=mask)")
    lines.append("    comp_idx = 1")

    for l in range(1, kernel_l_max + 1):
        lines.append("")
        lines.append(f"    if L_MAX >= {l}:")
        for i, expr in enumerate(_forward_components(l)):
            lines.append(
                f"        tl.store(out_ptr + offsets * out_stride_n + comp_idx + {i}, {_pycode(expr)}, mask=mask)"
            )
        lines.append(f"        comp_idx += {2 * l + 1}")

    return "\n".join(lines) + "\n"


def _emit_irreps_bwd_kernel(kernel_l_max: int) -> str:
    lines: List[str] = []
    lines.append('@triton.autotune(configs=_IRREPS_BWD_AUTOTUNE_CONFIGS, key=["L_MAX"])')
    lines.append("@triton.jit")
    lines.append("def irreps_bwd_kernel(")
    lines.append("    go_ptr,")
    lines.append("    x_ptr,")
    lines.append("    gx_ptr,")
    lines.append("    n_batch,")
    lines.append("    go_stride_n,")
    lines.append("    x_stride_n,")
    lines.append("    gx_stride_n,")
    lines.append("    L_MAX: tl.constexpr,")
    lines.append("    BLOCK_SIZE: tl.constexpr,")
    lines.append("):")
    lines.append('    """VJP: ``gx_k = sum_a g_a * dT_a/dx_k`` for ``k in (x, y, z)``."""')
    _emit_row_prologue(lines)
    _emit_vec3_load(lines, ("x", "y", "z"), "x_ptr", "x_stride_n")
    lines.append("    grad_x = tl.zeros_like(x)")
    lines.append("    grad_y = tl.zeros_like(y)")
    lines.append("    grad_z = tl.zeros_like(z)")
    lines.append("")
    lines.append("    # l = 0 has constant forward and contributes nothing to the gradient")
    lines.append("    comp_idx = 1")

    for l in range(1, kernel_l_max + 1):
        lines.append("")
        lines.append(f"    if L_MAX >= {l}:")
        n = 2 * l + 1
        for i in range(n):
            lines.append(
                f"        g{i} = tl.load(go_ptr + offsets * go_stride_n + comp_idx + {i}, mask=mask, other=0.0)"
            )
        gx, gy, gz = irreps_backward_sympy(l)
        if gx != 0:
            lines.append(f"        grad_x += {_pycode(gx)}")
        if gy != 0:
            lines.append(f"        grad_y += {_pycode(gy)}")
        if gz != 0:
            lines.append(f"        grad_z += {_pycode(gz)}")
        lines.append(f"        comp_idx += {n}")

    lines.append("")
    lines.append("    tl.store(gx_ptr + offsets * gx_stride_n + 0, grad_x, mask=mask)")
    lines.append("    tl.store(gx_ptr + offsets * gx_stride_n + 1, grad_y, mask=mask)")
    lines.append("    tl.store(gx_ptr + offsets * gx_stride_n + 2, grad_z, mask=mask)")

    return "\n".join(lines) + "\n"


def _emit_irreps_dbwd_kernel(kernel_l_max: int) -> str:
    lines: List[str] = []
    lines.append('@triton.autotune(configs=_IRREPS_DBWD_AUTOTUNE_CONFIGS, key=["L_MAX"])')
    lines.append("@triton.jit")
    lines.append("def irreps_dbwd_kernel(")
    lines.append("    go_ptr,")
    lines.append("    x_ptr,")
    lines.append("    v_ptr,")
    lines.append("    d_go_ptr,")
    lines.append("    d_x_ptr,")
    lines.append("    n_batch,")
    lines.append("    go_stride_n,")
    lines.append("    x_stride_n,")
    lines.append("    v_stride_n,")
    lines.append("    d_go_stride_n,")
    lines.append("    d_x_stride_n,")
    lines.append("    L_MAX: tl.constexpr,")
    lines.append("    BLOCK_SIZE: tl.constexpr,")
    lines.append("):")
    lines.append('    """VJP of the VJP with tangent ``v``: ``d_go_a = sum_k v_k * dT_a/dx_k``')
    lines.append('    (JVP) and ``d_x_k = d/dx_k (v^T gx)`` (bilinear in ``v``, ``g``)."""')
    _emit_row_prologue(lines)
    _emit_vec3_load(lines, ("x", "y", "z"), "x_ptr", "x_stride_n")
    _emit_vec3_load(lines, ("vx", "vy", "vz"), "v_ptr", "v_stride_n")
    lines.append("    d_x_x = tl.zeros_like(x)")
    lines.append("    d_x_y = tl.zeros_like(y)")
    lines.append("    d_x_z = tl.zeros_like(z)")
    lines.append("")
    lines.append("    # l = 0: forward is constant, d_go[0] = 0 and no d_x contribution")
    lines.append("    tl.store(d_go_ptr + offsets * d_go_stride_n + 0, tl.zeros_like(x), mask=mask)")
    lines.append("    comp_idx = 1")

    for l in range(1, kernel_l_max + 1):
        lines.append("")
        lines.append(f"    if L_MAX >= {l}:")
        n = 2 * l + 1
        for i in range(n):
            lines.append(
                f"        g{i} = tl.load(go_ptr + offsets * go_stride_n + comp_idx + {i}, mask=mask, other=0.0)"
            )
        dgrad_g, (dx, dy, dz) = irreps_double_backward_sympy(l)
        for i, expr in enumerate(dgrad_g):
            lines.append(
                f"        tl.store(d_go_ptr + offsets * d_go_stride_n + comp_idx + {i}, {_pycode(expr)}, mask=mask)"
            )
        if dx != 0:
            lines.append(f"        d_x_x += {_pycode(dx)}")
        if dy != 0:
            lines.append(f"        d_x_y += {_pycode(dy)}")
        if dz != 0:
            lines.append(f"        d_x_z += {_pycode(dz)}")
        lines.append(f"        comp_idx += {n}")

    lines.append("")
    lines.append("    tl.store(d_x_ptr + offsets * d_x_stride_n + 0, d_x_x, mask=mask)")
    lines.append("    tl.store(d_x_ptr + offsets * d_x_stride_n + 1, d_x_y, mask=mask)")
    lines.append("    tl.store(d_x_ptr + offsets * d_x_stride_n + 2, d_x_z, mask=mask)")

    return "\n".join(lines) + "\n"


def build_triton_irreps_module_source(kernel_l_max: int) -> str:
    """Generate a Triton Cartesian expansion module as Python source.

    Args:
        kernel_l_max (int): Maximum tensor rank included in the generated module.

    Returns:
        str: Python module source. The cache layer adds the fingerprint header
            when writing the module.
    """
    if kernel_l_max < 0:
        raise ValueError(f"kernel_l_max must be >= 0, got {kernel_l_max}.")
    pieces = [
        _MODULE_DOCSTRING + "\n",
        "\n",
        "import triton\n",
        "import triton.language as tl\n",
        "\n",
        f"KERNEL_L_MAX = {kernel_l_max}\n",
        "\n",
        _FWD_CONFIGS_SOURCE,
        "\n",
        _BWD_CONFIGS_SOURCE,
        "\n",
        _DBWD_CONFIGS_SOURCE,
        "\n\n",
        _emit_irreps_fwd_kernel(kernel_l_max),
        "\n\n",
        _emit_irreps_bwd_kernel(kernel_l_max),
        "\n\n",
        _emit_irreps_dbwd_kernel(kernel_l_max),
    ]
    return "".join(pieces)
