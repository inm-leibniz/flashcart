"""Generate natural-parity Cartesian tensor products and their derivatives.

The supported paths satisfy the triangle rule and require ``l1 + l2 - l_out`` to be
even. Pairwise index contractions preserve the natural parity ``(-1)**l`` of the
Cartesian tensors.

Implements Eq. (2) of Zaverkin et al., NeurIPS 2024. Stored inputs use the same ``2l+1``
basis from ``flashcart.o3._codegen_common.stored_basis``.

We follow the convention::

    TP_even(T1, T2)^{l_out} = N(l1, l2, l_out) * STF_{l_out}{Sym_{l_out}{
        contract_k(T1_full (x) T2_full)}}

where ``k = (l1 + l2 - l_out) / 2`` index pairs of the outer product are contracted,
``Sym_{l_out}`` is the normalised average over ``l_out!`` permutations, and
``STF_{l_out}`` is the symmetric-trace-free projection::

    STF_L(R)_{i1...iL} = sum_{k=0}^{floor(L/2)} c_k(L)
                         * Sym{delta^k (x) Tr^k(R)}_{i1...iL}

with ``c_k(L) = (-1)^k L! (2L-2k-1)!! / [k! (L-2k)! (2L-1)!! 2^k]``.

The closed-form factor ``C_{l1 l2 l3}`` is combined with a shuffle multiplicity
``binom(l_out, l1 - k)`` because the generator uses a fully normalised ``(1/l_out!)``
permutation average in ``_sym_average``, whereas the paper sums unique partitions only.
Hence ``N(l1, l2, l_out) = binom(...) * C`` via ``_even_tp_norm``.

For each allowed path ``(l1, l2, l_out)`` we build one forward expression
``f_a(S1, S2)`` (``a = 0...2*l_out``) and differentiate it in SymPy.

* backward (VJP) with respect to stored inputs ``S1``, ``S2`` (cotangent ``g`` on the output):

      grad_S1_b = sum_a g_a * d(f_a)/d(S1_b)
      grad_S2_b = sum_a g_a * d(f_a)/d(S2_b)

  At runtime the weighted map is ``out = w * TP_even(T1, T2)``. The VJP
  with respect to ``T1`` / ``T2`` therefore uses ``grad_out * w``. The backward kernels
  can evaluate both input gradients and ``grad_weights``. The Python code in
  ``flashcart.o3._tensor_product`` selects fused or separate kernel launches
  according to the requested outputs and configuration.

* double-backward (VJP of VJP) with input tangents ``vT1``, ``vT2`` and cotangent ``g``
  on the output:

      dgrad_g_a   = sum_b vT1_b * d(f_a)/d(S1_b) + sum_b vT2_b * d(f_a)/d(S2_b)
      dgrad_S1_b' = sum_{a,c} g_a * vT2_c * d^2 f_a / (d S1_b' d S2_c)
      dgrad_S2_b' = sum_{a,c} g_a * vT1_c * d^2 f_a / (d S1_c d S2_b')

  Because ``f`` is bilinear in ``(S1, S2)``, ``d^2 f / d S1^2`` and
  ``d^2 f / d S2^2`` vanish. Only the cross term contributes. For the
  weighted product, the derivative with respect to the incoming gradient is
  ``w * (J1 @ vT1 + J2 @ vT2) + vw * f(T1, T2)``, where ``J1`` and ``J2``
  are the two input Jacobians. The derivatives with respect to the inputs
  use the effective tangents ``w * vT2 + vw * T2`` and
  ``w * vT1 + vw * T1``, respectively. These input branches do not multiply
  the incoming gradient by ``w`` again.

``build_triton_tp_module_source`` emits forward, backward, and
double-backward kernels for independent rows and neighbor aggregation.
Each path uses the same symbolic expressions, including swapped-operand
variants when the input ranks differ.

The edge-parallel kernels process rows independently and accumulate
contributions to receiver nodes when needed. The receiver-grouped kernels
process incoming edges together. Their forward calculation accumulates
contributions before storing them. Derivative calculations use atomic
additions where several programs contribute to the same input gradient.

The Python code in ``flashcart.o3._tensor_product`` selects the requested derivative
outputs and can divide them across separate kernel launches. It also groups output
tensor ranks to limit the number of intermediate values held by a program. For
receivers with many incoming edges, backward work can be divided between groups of
edges. Rank, layout, and output-selection flags are compile-time constants.

Autotuning resets outputs that accumulate contributions between trials.
The launch helpers in ``flashcart.o3._tensor_product`` repeat grouped launches
when tuning clears partial results.
The generated names ``go`` and ``dgo`` refer to the incoming output
gradient and its derivative. ``uc`` denotes an unweighted contraction,
and ``uab`` denotes the input-tangent contribution. ``eff_*`` holds an
effective input tangent for a weighted derivative.
"""

import math
import re as _re
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product as _iter_product, permutations as _iter_perm
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import sympy as sp

from flashcart.o3._codegen_common import (
    _lambdify,
    _pycode,
    double_factorial,
    stored_basis,
)


@lru_cache(maxsize=None)
def _even_tp_norm(l1: int, l2: int, l_out: int) -> sp.Rational:
    """Compute the normalization for one natural-parity tensor-product path.

    The generator averages over all output-index permutations. The normalization
    includes the shuffle multiplicity needed to match the distinct-partition
    convention of Eq. (2) of Zaverkin et al., NeurIPS 2024.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        sympy.Rational: Exact normalization multiplying the projected product.

    Raises:
        ValueError: The ranks do not define an allowed natural-parity path.
    """
    if not even_tp_path_allowed(l1, l2, l_out):
        raise ValueError(f"Not an allowed even-TP path: ({l1}, {l2}, {l_out})")
    k = (l1 + l2 - l_out) // 2
    J = l1 + l2 + l_out
    J1, J2, J3 = J - 2 * l1 - 1, J - 2 * l2 - 1, J - 2 * l_out - 1
    num = (
        math.factorial(l1)
        * math.factorial(l2)
        * double_factorial(2 * l_out - 1)
        * math.factorial((J1 + 1) // 2)
        * math.factorial((J2 + 1) // 2)
    )
    den = (
        math.factorial(l_out)
        * double_factorial(J1)
        * double_factorial(J2)
        * double_factorial(J3)
        * math.factorial(J // 2)
    )
    ictp_norm = sp.Rational(num, den)
    shuffle = math.factorial(l_out) // (math.factorial(l1 - k) * math.factorial(l2 - k))
    return shuffle * ictp_norm


def even_tp_path_allowed(l1: int, l2: int, l_out: int) -> bool:
    """Check the triangle and natural-parity rules for one coupling path.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        bool: True for nonnegative ranks satisfying the triangle rule and an
            even sum ``l1 + l2 + l_out``.
    """
    if l1 < 0 or l2 < 0 or l_out < 0:
        return False
    if l_out < abs(l1 - l2) or l_out > l1 + l2:
        return False
    if (l1 + l2 + l_out) % 2 != 0:
        return False
    return True


def even_tp_allowed_paths(l1_max: int, l2_max: int, l_out_max: int) -> List[Tuple[int, int, int]]:
    """Enumerate allowed natural-parity paths within the requested rank limits.

    Args:
        l1_max (int): Maximum first-input tensor rank.
        l2_max (int): Maximum second-input tensor rank.
        l_out_max (int): Maximum output tensor rank.

    Returns:
        list[tuple[int, int, int]]: ``(l1, l2, l_out)`` paths ordered first by
            ``l_out``, then ``l1``, then ``l2``. This enumeration is not the
            canonical-then-swapped ordering used for runtime path weights.
    """
    paths: List[Tuple[int, int, int]] = []
    for l_out in range(l_out_max + 1):
        for l1 in range(l1_max + 1):
            for l2 in range(l2_max + 1):
                if even_tp_path_allowed(l1, l2, l_out):
                    paths.append((l1, l2, l_out))
    return paths


@lru_cache(maxsize=None)
def even_tp_t1_symbols(l1: int) -> Tuple[sp.Symbol, ...]:
    n = 2 * l1 + 1
    return tuple(sp.symbols(f"S1_0:{n}", real=True))


@lru_cache(maxsize=None)
def even_tp_t2_symbols(l2: int) -> Tuple[sp.Symbol, ...]:
    n = 2 * l2 + 1
    return tuple(sp.symbols(f"S2_0:{n}", real=True))


@lru_cache(maxsize=None)
def even_tp_g_symbols(l_out: int) -> Tuple[sp.Symbol, ...]:
    n = 2 * l_out + 1
    return tuple(sp.symbols(f"gO_0:{n}", real=True))


@lru_cache(maxsize=None)
def even_tp_vt1_symbols(l1: int) -> Tuple[sp.Symbol, ...]:
    n = 2 * l1 + 1
    return tuple(sp.symbols(f"vT1_0:{n}", real=True))


@lru_cache(maxsize=None)
def even_tp_vt2_symbols(l2: int) -> Tuple[sp.Symbol, ...]:
    n = 2 * l2 + 1
    return tuple(sp.symbols(f"vT2_0:{n}", real=True))


def _full_cartesian_from_stored(l: int, stored_syms: Sequence[sp.Expr]) -> Dict[Tuple[int, ...], sp.Expr]:
    """Reconstruct a full symmetric traceless tensor symbolically.

    Args:
        l (int): Tensor rank.
        stored_syms (Sequence[sympy.Expr]): Independent components in
            ``stored_basis(l)`` order.

    Returns:
        dict[tuple[int, ...], sympy.Expr]: Expression for every Cartesian index
            tuple. Components containing at least two z indices are determined
            recursively by tracelessness.
    """
    if l == 0:
        return {(): stored_syms[0]}
    basis = stored_basis(l)
    basis_index = {tup: i for i, tup in enumerate(basis)}
    cache: Dict[Tuple[int, int, int], sp.Expr] = {}

    def get_sorted(a: int, b: int, c: int) -> sp.Expr:
        if (a, b, c) in cache:
            return cache[(a, b, c)]
        if (a, b, c) in basis_index:
            val: sp.Expr = stored_syms[basis_index[(a, b, c)]]
        elif c >= 2:
            val = -get_sorted(a + 2, b, c - 2) - get_sorted(a, b + 2, c - 2)
        else:
            raise ValueError(f"Unexpected multi-index ({a}, {b}, {c}) for l={l}")
        cache[(a, b, c)] = val
        return val

    full: Dict[Tuple[int, ...], sp.Expr] = {}
    for full_idx in _iter_product(range(3), repeat=l):
        a = full_idx.count(0)
        b = full_idx.count(1)
        c = full_idx.count(2)
        full[full_idx] = get_sorted(a, b, c)
    return full


def _outer_product(
    T1: Dict[Tuple[int, ...], sp.Expr],
    l1: int,
    T2: Dict[Tuple[int, ...], sp.Expr],
    l2: int,
) -> Dict[Tuple[int, ...], sp.Expr]:
    F: Dict[Tuple[int, ...], sp.Expr] = {}
    if l1 == 0 and l2 == 0:
        F[()] = T1[()] * T2[()]
        return F
    if l1 == 0:
        for idx in _iter_product(range(3), repeat=l2):
            F[idx] = T1[()] * T2[idx]
        return F
    if l2 == 0:
        for idx in _iter_product(range(3), repeat=l1):
            F[idx] = T1[idx] * T2[()]
        return F
    for idx1 in _iter_product(range(3), repeat=l1):
        for idx2 in _iter_product(range(3), repeat=l2):
            F[idx1 + idx2] = T1[idx1] * T2[idx2]
    return F


def _contract_k_pairs(F: Dict[Tuple[int, ...], sp.Expr], l1: int, l2: int, k: int) -> Dict[Tuple[int, ...], sp.Expr]:
    if k == 0:
        return F
    L_out = l1 + l2 - 2 * k
    C: Dict[Tuple[int, ...], sp.Expr] = {}
    for out_idx in _iter_product(range(3), repeat=L_out):
        prefix = out_idx[: l1 - k]
        suffix = out_idx[l1 - k :]
        expr: sp.Expr = sp.Integer(0)
        for c_idx in _iter_product(range(3), repeat=k):
            full_idx = prefix + c_idx + c_idx + suffix
            expr += F[full_idx]
        C[out_idx] = expr
    return C


def _sym_average(R: Dict[Tuple[int, ...], sp.Expr], L: int) -> Dict[Tuple[int, ...], sp.Expr]:
    """Average a tensor over all permutations of its Cartesian indices.

    Args:
        R (dict[tuple[int, ...], sympy.Expr]): Full Cartesian tensor components.
        L (int): Tensor rank.

    Returns:
        dict[tuple[int, ...], sympy.Expr]: Symmetrized components, divided by
            ``L!``. Repeated index permutations retain their multiplicities.
    """
    if L <= 1:
        return {idx: R[idx] for idx in R}
    Sym: Dict[Tuple[int, ...], sp.Expr] = {}
    L_fact = math.factorial(L)
    for idx in _iter_product(range(3), repeat=L):
        expr: sp.Expr = sp.Integer(0)
        for perm in _iter_perm(range(L)):
            permuted = tuple(idx[perm[i]] for i in range(L))
            expr += R[permuted]
        Sym[idx] = sp.Rational(1, L_fact) * expr
    return Sym


def _trace_k(R: Dict[Tuple[int, ...], sp.Expr], L: int, k: int) -> Dict[Tuple[int, ...], sp.Expr]:
    if k == 0:
        return R
    L_curr = L
    R_curr = R
    for _ in range(k):
        new_L = L_curr - 2
        new_R: Dict[Tuple[int, ...], sp.Expr] = {}
        for idx_rest in _iter_product(range(3), repeat=new_L):
            expr: sp.Expr = sp.Integer(0)
            for a in range(3):
                expr += R_curr[(a, a) + idx_rest]
            new_R[idx_rest] = expr
        R_curr = new_R
        L_curr = new_L
    return R_curr


def _sym_delta_k_R(R_trace: Dict[Tuple[int, ...], sp.Expr], L_full: int, k: int) -> Dict[Tuple[int, ...], sp.Expr]:
    L_rest = L_full - 2 * k
    L_fact = math.factorial(L_full)
    Sym: Dict[Tuple[int, ...], sp.Expr] = {}
    for idx in _iter_product(range(3), repeat=L_full):
        expr: sp.Expr = sp.Integer(0)
        for perm in _iter_perm(range(L_full)):
            permuted = tuple(idx[perm[i]] for i in range(L_full))
            # delta_{p0,p1} * delta_{p2,p3} * ... * delta_{p_{2k-2},p_{2k-1}}
            ok = True
            for j in range(k):
                if permuted[2 * j] != permuted[2 * j + 1]:
                    ok = False
                    break
            if not ok:
                continue
            rest_key = permuted[2 * k :] if L_rest > 0 else ()
            expr += R_trace[rest_key]
        Sym[idx] = sp.Rational(1, L_fact) * expr
    return Sym


def _stf_projection(R: Dict[Tuple[int, ...], sp.Expr], L: int) -> Dict[Tuple[int, ...], sp.Expr]:
    """Remove traces from a symmetric Cartesian tensor.

    Args:
        R (dict[tuple[int, ...], sympy.Expr]): Symmetric tensor components.
            Symmetry is assumed, not checked or imposed here.
        L (int): Tensor rank.

    Returns:
        dict[tuple[int, ...], sympy.Expr]: Symmetric traceless components with
            all polynomial expressions expanded.
    """
    if L < 2:
        return {idx: sp.expand(R[idx]) for idx in R}
    result = {idx: R[idx] for idx in R}
    denom = double_factorial(2 * L - 1)
    L_fact = math.factorial(L)
    for k in range(1, L // 2 + 1):
        sign = (-1) ** k
        num = L_fact * double_factorial(2 * L - 2 * k - 1)
        denom_k = math.factorial(k) * math.factorial(L - 2 * k) * denom * (2**k)
        c_k = sp.Rational(sign * num, denom_k)
        Tr_k = _trace_k(R, L, k)
        Sym_k = _sym_delta_k_R(Tr_k, L, k)
        for idx in result:
            result[idx] = result[idx] + c_k * Sym_k[idx]
    return {idx: sp.expand(result[idx]) for idx in result}


def _stored_from_full(R: Dict[Tuple[int, ...], sp.Expr], l: int) -> List[sp.Expr]:
    components: List[sp.Expr] = []
    for a, b, c in stored_basis(l):
        idx = tuple([0] * a + [1] * b + [2] * c)
        components.append(sp.expand(R[idx]))
    return components


@lru_cache(maxsize=None)
def _even_tp_forward_components(l1: int, l2: int, l_out: int) -> Tuple[sp.Expr, ...]:
    if not even_tp_path_allowed(l1, l2, l_out):
        raise ValueError(f"Not an allowed even-TP path: ({l1}, {l2}, {l_out})")
    S1 = even_tp_t1_symbols(l1)
    S2 = even_tp_t2_symbols(l2)
    T1 = _full_cartesian_from_stored(l1, S1)
    T2 = _full_cartesian_from_stored(l2, S2)
    F = _outer_product(T1, l1, T2, l2)
    k = (l1 + l2 - l_out) // 2
    C = _contract_k_pairs(F, l1, l2, k)
    Sym_C = _sym_average(C, l_out)
    STF_C = _stf_projection(Sym_C, l_out)
    stored = _stored_from_full(STF_C, l_out)
    norm = _even_tp_norm(l1, l2, l_out)
    return tuple(sp.expand(norm * e) for e in stored)


def even_tp_forward_sympy(l1: int, l2: int, l_out: int) -> List[sp.Expr]:
    """Construct independent component expressions for an unweighted tensor product.

    Weight derivatives are added separately by the kernel generator.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        list[sympy.Expr]: The ``2*l_out + 1`` output polynomials in stored
            component order.
    """
    return list(_even_tp_forward_components(l1, l2, l_out))


def even_tp_backward_sympy(l1: int, l2: int, l_out: int) -> Tuple[List[sp.Expr], List[sp.Expr]]:
    """Construct input-gradient expressions for an unweighted tensor product.

    Weight derivatives are added separately by the kernel generator.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        tuple[list[sympy.Expr], list[sympy.Expr]]: Gradients with respect to
            the first and second inputs after contraction with the incoming
            output gradient. Each list follows its input stored-component order.
    """
    S1 = even_tp_t1_symbols(l1)
    S2 = even_tp_t2_symbols(l2)
    g = even_tp_g_symbols(l_out)
    forward = _even_tp_forward_components(l1, l2, l_out)
    n_out = 2 * l_out + 1

    grad_T1: List[sp.Expr] = []
    for b in range(2 * l1 + 1):
        expr = sum(g[a] * sp.diff(forward[a], S1[b]) for a in range(n_out))
        grad_T1.append(sp.expand(expr))

    grad_T2: List[sp.Expr] = []
    for b in range(2 * l2 + 1):
        expr = sum(g[a] * sp.diff(forward[a], S2[b]) for a in range(n_out))
        grad_T2.append(sp.expand(expr))

    return grad_T1, grad_T2


def even_tp_double_backward_sympy(l1: int, l2: int, l_out: int) -> Tuple[List[sp.Expr], List[sp.Expr], List[sp.Expr]]:
    # dgrad_g_a = sum_b vT1_b d f_a/d S1_b + sum_b vT2_b d f_a/d S2_b
    # dgrad_T1_b' = sum_{a, b} g_a vT2_b  d^2 f_a / d S1_b' d S2_b
    # dgrad_T2_b' = sum_{a, b} g_a vT1_b  d^2 f_a / d S1_b  d S2_b'
    """Differentiate the input gradients of an unweighted tensor product.

    Weight derivatives are added separately by the kernel generator.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        tuple[list[sympy.Expr], list[sympy.Expr], list[sympy.Expr]]:
            Derivatives with respect to the incoming output gradient, first
            input, and second input, respectively. Each list follows the
            stored-component order of its tensor rank.
    """
    S1 = even_tp_t1_symbols(l1)
    S2 = even_tp_t2_symbols(l2)
    g = even_tp_g_symbols(l_out)
    vT1 = even_tp_vt1_symbols(l1)
    vT2 = even_tp_vt2_symbols(l2)
    forward = _even_tp_forward_components(l1, l2, l_out)
    n_out = 2 * l_out + 1
    n1 = 2 * l1 + 1
    n2 = 2 * l2 + 1

    dgrad_g: List[sp.Expr] = []
    for a in range(n_out):
        expr = sum(vT1[b] * sp.diff(forward[a], S1[b]) for b in range(n1))
        expr += sum(vT2[b] * sp.diff(forward[a], S2[b]) for b in range(n2))
        dgrad_g.append(sp.expand(expr))

    dgrad_T1: List[sp.Expr] = []
    for bp in range(n1):
        expr = sp.Integer(0)
        for a in range(n_out):
            for b in range(n2):
                expr += g[a] * vT2[b] * sp.diff(forward[a], S1[bp], S2[b])
        dgrad_T1.append(sp.expand(expr))

    dgrad_T2: List[sp.Expr] = []
    for bp in range(n2):
        expr = sp.Integer(0)
        for a in range(n_out):
            for b in range(n1):
                expr += g[a] * vT1[b] * sp.diff(forward[a], S1[b], S2[bp])
        dgrad_T2.append(sp.expand(expr))

    return dgrad_g, dgrad_T1, dgrad_T2


@lru_cache(maxsize=None)
def compile_even_tp_forward(l1: int, l2: int, l_out: int):
    """Build a cached numerical reference callable for an unweighted path.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        Callable: Function accepting the first and second stored inputs, with
            each component passed as a separate argument. The result lists
            the stored output components.
    """
    S1 = even_tp_t1_symbols(l1)
    S2 = even_tp_t2_symbols(l2)
    return _lambdify((*S1, *S2), list(_even_tp_forward_components(l1, l2, l_out)))


@lru_cache(maxsize=None)
def compile_even_tp_backward(l1: int, l2: int, l_out: int):
    """Build a cached numerical reference callable for an unweighted path.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        Callable: Function accepting the two stored inputs followed by the
            stored output gradient, with each component passed as a separate
            argument. The result is one flattened list of first-input and
            second-input gradients, in that order.
    """
    S1 = even_tp_t1_symbols(l1)
    S2 = even_tp_t2_symbols(l2)
    g = even_tp_g_symbols(l_out)
    grad_T1, grad_T2 = even_tp_backward_sympy(l1, l2, l_out)
    return _lambdify((*S1, *S2, *g), [*grad_T1, *grad_T2])


@lru_cache(maxsize=None)
def compile_even_tp_double_backward(l1: int, l2: int, l_out: int):
    """Build a cached numerical reference callable for an unweighted path.

    Args:
        l1 (int): First-input tensor rank.
        l2 (int): Second-input tensor rank.
        l_out (int): Output tensor rank.

    Returns:
        Callable: Function accepting the two stored inputs, the stored output gradient, then the
            two input-gradient seeds, with each component
            passed as a separate argument. The result is one flattened list of
            derivatives with respect to the output gradient, first input,
            and second input, in that order.
    """
    S1 = even_tp_t1_symbols(l1)
    S2 = even_tp_t2_symbols(l2)
    g = even_tp_g_symbols(l_out)
    vT1 = even_tp_vt1_symbols(l1)
    vT2 = even_tp_vt2_symbols(l2)
    dgrad_g, dgrad_T1, dgrad_T2 = even_tp_double_backward_sympy(l1, l2, l_out)
    return _lambdify(
        (*S1, *S2, *g, *vT1, *vT2),
        [*dgrad_g, *dgrad_T1, *dgrad_T2],
    )


# ----------------------------------
# Codegen for PyTorch tensor-product
# ----------------------------------


def _enumerate_paths(
    in1_l_max: int,
    in2_l_max: int,
    out_l_max: int,
    symmetric_product: bool,
) -> List[Tuple[int, int, int, bool]]:
    paths: List[Tuple[int, int, int, bool]] = []
    l_pair_max = max(in1_l_max, in2_l_max)
    for l_out in range(out_l_max + 1):
        for l1 in range(l_pair_max + 1):
            for l2 in range(l1 + 1):
                if not even_tp_path_allowed(l1, l2, l_out):
                    continue
                if l1 > in1_l_max or l2 > in2_l_max:
                    continue
                paths.append((l1, l2, l_out, False))
        if not symmetric_product:
            for l1 in range(l_pair_max + 1):
                for l2 in range(l1 + 1):
                    if l1 == l2:
                        continue
                    if not even_tp_path_allowed(l1, l2, l_out):
                        continue
                    if l1 > in2_l_max or l2 > in1_l_max:
                        continue
                    paths.append((l1, l2, l_out, True))
    return paths


def _emit_stored_loads(side: str, l_max: int, features: int, indent: str = "        ") -> List[str]:
    in_name = "in1" if side == "S1" else "in2"
    lines: List[str] = []
    offset = 0
    for l in range(l_max + 1):
        size = (2 * l + 1) * features
        if l == 0:
            lines.append(f"{indent}_{side}_l0_0 = {in_name}[:, {offset}:{offset + size}]")
        else:
            block = f"_{in_name}_l{l}"
            lines.append(
                f"{indent}{block} = {in_name}[:, {offset}:{offset + size}]"
                f".reshape(n_batch, {2 * l + 1}, {features})"
            )
            for i in range(2 * l + 1):
                lines.append(f"{indent}_{side}_l{l}_{i} = {block}[:, {i}]")
        offset += size
    return lines


def _path_substitutions(l1: int, l2: int, is_swap: bool) -> Dict[sp.Symbol, sp.Symbol]:
    s1_sym = even_tp_t1_symbols(l1)
    s2_sym = even_tp_t2_symbols(l2)
    s1_dst = "S2" if is_swap else "S1"
    s2_dst = "S1" if is_swap else "S2"
    subs: Dict[sp.Symbol, sp.Symbol] = {}
    for i, s in enumerate(s1_sym):
        subs[s] = sp.Symbol(f"_{s1_dst}_l{l1}_{i}", real=True)
    for j, s in enumerate(s2_sym):
        subs[s] = sp.Symbol(f"_{s2_dst}_l{l2}_{j}", real=True)
    return subs


def build_py_tp_forward_source(
    in1_l_max: int,
    in2_l_max: int,
    out_l_max: int,
    in1_features: int,
    in2_features: int,
    symmetric_product: bool,
    shared_weights: bool,
    reduce_paths: bool,
) -> str:
    """Generate a PyTorch tensor product specialized to its layout.

    The source defines a callable module for independent rows or edge aggregation.
    Weights and outputs follow the layout used by ``TensorProduct.forward``.

    Args:
        in1_l_max (int): Maximum first-input tensor rank.
        in2_l_max (int): Maximum second-input tensor rank.
        out_l_max (int): Maximum output tensor rank.
        in1_features (int): First-input feature channels.
        in2_features (int): Second-input feature channels.
        symmetric_product (bool): Retain only couplings with ``l1 >= l2``.
        shared_weights (bool): Use one path-weight vector for every row or edge.
        reduce_paths (bool): Sum path contributions within each output rank.

    Returns:
        str: Python class source defining ``PyTensorProductForward``.
    """
    out_features = max(in1_features, in2_features)
    paths = _enumerate_paths(in1_l_max, in2_l_max, out_l_max, symmetric_product)

    out_buckets: Dict[int, List[str]] = {l: [] for l in range(out_l_max + 1)}
    lines: List[str] = []
    lines.append("class PyTensorProductForward(torch.nn.Module):")
    lines.append("    def forward(self, in1, in2, weights, idx_i, idx_j):")
    lines.append("        needs_scatter = idx_i is not None and idx_j is not None")
    lines.append("        in1_batch = in1.shape[0]")
    lines.append("        if needs_scatter:")
    lines.append("            in1 = in1.index_select(0, idx_j)")
    lines.append("        n_batch = in1.shape[0]")

    lines.extend(_emit_stored_loads("S1", in1_l_max, in1_features))
    lines.extend(_emit_stored_loads("S2", in2_l_max, in2_features))

    for path_idx, (l1, l2, l_out, is_swap) in enumerate(paths):
        exprs = even_tp_forward_sympy(l1, l2, l_out)
        subs = _path_substitutions(l1, l2, is_swap)
        comp_names: List[str] = []
        for i, expr in enumerate(exprs):
            name = f"_p{path_idx}_c{i}"
            comp_names.append(name)
            lines.append(f"        {name} = {_pycode(expr.xreplace(subs))}")

        weight_start = path_idx * out_features
        weight_end = weight_start + out_features
        if shared_weights:
            w_expr = f"weights[{weight_start}:{weight_end}]"
        else:
            w_expr = f"weights[:, {weight_start}:{weight_end}]"

        weighted = f"_p{path_idx}_w"
        if l_out == 0:
            lines.append(f"        {weighted} = {comp_names[0]} * {w_expr}")
        else:
            stack_args = ", ".join(comp_names)
            stacked = f"_p{path_idx}_s"
            lines.append(f"        {stacked} = torch.stack([{stack_args}], dim=1)")
            if shared_weights:
                lines.append(f"        {weighted} = {stacked} * {w_expr}")
            else:
                lines.append(f"        {weighted} = {stacked} * {w_expr}.unsqueeze(1)")
        out_buckets[l_out].append(weighted)

    out_l_flats: List[str] = []
    for l_out in range(out_l_max + 1):
        bucket = out_buckets[l_out]
        flat_name = f"_out_l{l_out}_flat"
        out_l_flats.append(flat_name)
        if not bucket:
            lines.append(f"        {flat_name} = in1.new_zeros(n_batch, 0)")
            continue
        if l_out == 0:
            if reduce_paths:
                lines.append(f"        {flat_name} = {' + '.join(bucket)}")
            elif len(bucket) == 1:
                lines.append(f"        {flat_name} = {bucket[0]}")
            else:
                lines.append(
                    f"        {flat_name} = torch.stack([{', '.join(bucket)}], dim=1)" f".reshape(n_batch, -1)"
                )
        else:
            if reduce_paths:
                if len(bucket) == 1:
                    lines.append(f"        {flat_name} = {bucket[0]}.reshape(n_batch, -1)")
                else:
                    lines.append(f"        {flat_name} = ({' + '.join(bucket)})" f".reshape(n_batch, -1)")
            elif len(bucket) == 1:
                lines.append(f"        {flat_name} = {bucket[0]}.reshape(n_batch, -1)")
            else:
                lines.append(
                    f"        {flat_name} = torch.stack([{', '.join(bucket)}], dim=2)" f".reshape(n_batch, -1)"
                )

    if len(out_l_flats) == 1:
        lines.append(f"        out = {out_l_flats[0]}")
    else:
        lines.append(f"        out = torch.cat([{', '.join(out_l_flats)}], dim=-1)")

    lines.append("        if needs_scatter:")
    lines.append("            out = scatter_sum(out, idx_i, dim=0, dim_size=in1_batch)")
    lines.append("        return out")

    return "\n".join(lines) + "\n"


def _py_contract_function_name(l1: int, l2: int, l_out: int) -> str:
    return f"py_contract_l{l1}_l{l2}_to_l{l_out}"


def _emit_py_contract_function(l1: int, l2: int, l_out: int) -> str:
    if not even_tp_path_allowed(l1, l2, l_out):
        raise ValueError(f"Not an allowed even-TP path: ({l1}, {l2}, {l_out})")

    in1_names = [f"S1_l{l1}_{i}" for i in range(2 * l1 + 1)]
    in2_names = [f"S2_l{l2}_{i}" for i in range(2 * l2 + 1)]
    subs = {
        **{sym: sp.Symbol(name, real=True) for sym, name in zip(even_tp_t1_symbols(l1), in1_names)},
        **{sym: sp.Symbol(name, real=True) for sym, name in zip(even_tp_t2_symbols(l2), in2_names)},
    }

    lines: List[str] = []
    args = ", ".join(in1_names + in2_names)
    lines.append(f"def {_py_contract_function_name(l1, l2, l_out)}({args}):")

    out_names: List[str] = []
    for i, expr in enumerate(even_tp_forward_sympy(l1, l2, l_out)):
        name = f"c{i}"
        out_names.append(name)
        lines.append(f"    {name} = {_pycode(expr.xreplace(subs))}")
    if len(out_names) == 1:
        lines.append(f"    return {out_names[0]}")
    else:
        lines.append(f"    return {', '.join(out_names)}")
    return "\n".join(lines)


def _emit_py_forward_path(
    l1: int,
    l2: int,
    l_out: int,
    is_swap: bool,
    indent: str = "        ",
) -> List[str]:
    if is_swap:
        args = f"*in2_components[{l1}], *in1_components[{l2}]"
    else:
        args = f"*in1_components[{l1}], *in2_components[{l2}]"

    lines = [
        f"{indent}contract = {_py_contract_function_name(l1, l2, l_out)}({args})",
    ]
    if l_out != 0:
        lines.append(f"{indent}contract = torch.stack(contract, dim=1)")
    lines.extend(
        [
            f"{indent}weighted = apply_path_weight(",
            f"{indent}    contract, weights, path_idx, out_features, shared_weights",
            f"{indent})",
            f"{indent}out_lists[{l_out}].append(weighted)",
            f"{indent}path_idx += 1",
        ]
    )
    return lines


def build_py_tensor_product_forward_module_source(kernel_l_max: int) -> str:
    """Generate a PyTorch tensor-product module as Python source.

    Args:
        kernel_l_max (int): Maximum tensor rank included in the generated module.

    Returns:
        str: Python module source. The cache layer adds the fingerprint header
            when writing the module.
    """
    lines: List[str] = [
        '"""Auto-generated by flashcart.o3._codegen_tensor_product.build_py_tensor_product_forward_module_source."""',
        "",
        "import torch",
        "",
        "from flashcart.o3.utils import apply_path_weight",
        "from flashcart.utils.scatter import scatter_sum",
        "",
        f"KERNEL_L_MAX = {kernel_l_max}",
        "",
    ]

    helper_keys: List[Tuple[int, int, int]] = []
    for l_out in range(kernel_l_max + 1):
        for l1 in range(kernel_l_max + 1):
            for l2 in range(l1 + 1):
                if even_tp_path_allowed(l1, l2, l_out):
                    helper_keys.append((l1, l2, l_out))

    for key in helper_keys:
        lines.append(_emit_py_contract_function(*key))
        lines.append("")

    lines.extend(
        [
            "def _load_l(x, slices, l: int, n_batch: int, features: int):",
            "    start, stop = slices[l]",
            "    block = x[:, start:stop]",
            "    if l == 0:",
            "        return (block,)",
            "    dim_l = 2 * l + 1",
            "    block = block.reshape(n_batch, dim_l, features)",
            "    return tuple(block[:, i] for i in range(dim_l))",
            "",
            "",
            "def py_tensor_product_forward(",
            "    in1,",
            "    in2,",
            "    weights,",
            "    idx_i,",
            "    idx_j,",
            "    in1_l_max: int,",
            "    in2_l_max: int,",
            "    out_l_max: int,",
            "    in1_features: int,",
            "    in2_features: int,",
            "    in1_slices: list,",
            "    in2_slices: list,",
            "    symmetric_product: bool,",
            "    shared_weights: bool,",
            "    n_paths: list,",
            "    reduce_paths: bool,",
            "):",
            "    in1_l_max = int(in1_l_max)",
            "    in2_l_max = int(in2_l_max)",
            "    out_l_max = int(out_l_max)",
            "    if max(in1_l_max, in2_l_max, out_l_max) > KERNEL_L_MAX:",
            "        raise NotImplementedError(",
            '            f"Generated py_tensor_product supports l_max <= {KERNEL_L_MAX}; "',
            '            f"got in1_l_max={in1_l_max}, in2_l_max={in2_l_max}, "',
            '            f"out_l_max={out_l_max}."',
            "        )",
            "    in1_features = int(in1_features)",
            "    in2_features = int(in2_features)",
            "    if not (",
            "        in1_features == in2_features or in1_features == 1 or in2_features == 1",
            "    ):",
            "        raise ValueError(",
            '            f"Input dimensions must match unless one of them is 1. "',
            '            f"Got in1_features={in1_features} and in2_features={in2_features}."',
            "        )",
            "    in1_batch = in1.shape[0]",
            "    in2_batch = in2.shape[0]",
            "    needs_scatter = idx_i is not None and idx_j is not None",
            "    if in1_batch != in2_batch and not needs_scatter:",
            "        raise ValueError(",
            '            "idx_i and idx_j must be provided when in1 and in2 have different "',
            '            f"batch sizes. Got in1_batch={in1_batch}, in2_batch={in2_batch}."',
            "        )",
            "    if needs_scatter:",
            "        in1 = in1.index_select(0, idx_j)",
            "    n_batch = in1.shape[0]",
            "    out_features = max(in1_features, in2_features)",
            "    n_total_paths = sum(n_paths)",
            "    expected_weight_size = n_total_paths * out_features",
            "    shared_weights = bool(shared_weights)",
            "    reduce_paths = bool(reduce_paths)",
            "    if shared_weights:",
            "        if weights.dim() != 1:",
            '            raise ValueError(f"Shared weights must be 1D. Got shape {weights.shape}.")',
            "        if weights.shape[0] != expected_weight_size:",
            "            raise ValueError(",
            '                f"Shared weights must have size {expected_weight_size}, "',
            '                f"got {weights.shape[0]}."',
            "            )",
            "    else:",
            "        if weights.dim() != 2:",
            '            raise ValueError(f"Non-shared weights must be 2D. Got shape {weights.shape}.")',
            "        if weights.shape[0] != n_batch:",
            "            raise ValueError(",
            '                f"Non-shared weights must have weights.shape[0] = {n_batch}, "',
            '                f"got {weights.shape[0]}."',
            "            )",
            "        if weights.shape[1] != expected_weight_size:",
            "            raise ValueError(",
            '                f"Non-shared weights must have weights.shape[1] = {expected_weight_size}, "',
            '                f"got {weights.shape[1]}."',
            "            )",
            "    in1_components = [",
            "        _load_l(in1, in1_slices, l, n_batch, in1_features)",
            "        for l in range(in1_l_max + 1)",
            "    ]",
            "    in2_components = [",
            "        _load_l(in2, in2_slices, l, n_batch, in2_features)",
            "        for l in range(in2_l_max + 1)",
            "    ]",
            "    out_lists = [[] for _ in range(out_l_max + 1)]",
            "    path_idx = 0",
        ]
    )

    for l_out in range(kernel_l_max + 1):
        for l1 in range(kernel_l_max + 1):
            for l2 in range(l1 + 1):
                if not even_tp_path_allowed(l1, l2, l_out):
                    continue
                lines.append(f"    if out_l_max >= {l_out} and in1_l_max >= {l1} and in2_l_max >= {l2}:")
                lines.extend(_emit_py_forward_path(l1, l2, l_out, is_swap=False))
        for l1 in range(kernel_l_max + 1):
            for l2 in range(l1 + 1):
                if l1 == l2:
                    continue
                if not even_tp_path_allowed(l1, l2, l_out):
                    continue
                lines.append(
                    "    if ("
                    f"not symmetric_product and out_l_max >= {l_out} "
                    f"and in2_l_max >= {l1} and in1_l_max >= {l2}"
                    "):"
                )
                lines.extend(_emit_py_forward_path(l1, l2, l_out, is_swap=True))

    lines.extend(
        [
            "    out_flats = []",
            "    for l_out, bucket in enumerate(out_lists):",
            "        if not bucket:",
            "            out_flats.append(in1.new_empty(n_batch, 0))",
            "        elif l_out == 0:",
            "            if reduce_paths:",
            "                out_flats.append(sum(bucket))",
            "            elif len(bucket) == 1:",
            "                out_flats.append(bucket[0])",
            "            else:",
            "                out_flats.append(torch.stack(bucket, dim=1).reshape(n_batch, -1))",
            "        else:",
            "            if reduce_paths:",
            "                out_l = bucket[0] if len(bucket) == 1 else sum(bucket)",
            "                out_flats.append(out_l.reshape(n_batch, -1))",
            "            elif len(bucket) == 1:",
            "                out_flats.append(bucket[0].reshape(n_batch, -1))",
            "            else:",
            "                out_flats.append(torch.stack(bucket, dim=2).reshape(n_batch, -1))",
            "    out = out_flats[0] if len(out_flats) == 1 else torch.cat(out_flats, dim=-1)",
            "    if needs_scatter:",
            "        out = scatter_sum(out, idx_i, dim=0, dim_size=in1_batch)",
            "    return out",
        ]
    )
    return "\n".join(lines)


@lru_cache(maxsize=None)
def compile_py_tensor_product_forward(
    in1_l_max: int,
    in2_l_max: int,
    out_l_max: int,
    in1_features: int,
    in2_features: int,
    symmetric_product: bool,
    shared_weights: bool,
    reduce_paths: bool,
):
    """Build a PyTorch tensor-product callable for a fixed configuration.

    The callable is cached by all layout and operation settings. It supplies the
    fallback for ranks above the imported generated module's limit.

    Args:
        in1_l_max (int): Maximum first-input tensor rank.
        in2_l_max (int): Maximum second-input tensor rank.
        out_l_max (int): Maximum output tensor rank.
        in1_features (int): First-input feature channels.
        in2_features (int): Second-input feature channels.
        symmetric_product (bool): Retain only couplings with ``l1 >= l2``.
        shared_weights (bool): Use one path-weight vector for every row or edge.
        reduce_paths (bool): Sum path contributions within each output rank.

    Returns:
        Callable: Function accepting ``in1``, ``in2``, ``weights``, ``idx_i``,
            and ``idx_j`` and returning packed output features. Input, weight,
            index, and output layouts match ``TensorProduct.forward``.
    """
    import torch  # local import since codegen should not require torch at import time
    from flashcart.utils.scatter import scatter_sum

    src = build_py_tp_forward_source(
        in1_l_max=in1_l_max,
        in2_l_max=in2_l_max,
        out_l_max=out_l_max,
        in1_features=in1_features,
        in2_features=in2_features,
        symmetric_product=symmetric_product,
        shared_weights=shared_weights,
        reduce_paths=reduce_paths,
    )
    namespace: dict = {"torch": torch, "scatter_sum": scatter_sum}
    exec(src, namespace)  # noqa: S102 — trusted SymPy-emitted Python source
    return namespace["PyTensorProductForward"]()


# ----------------------------------
# Codegen for Triton tensor-product
# ----------------------------------


def _cartesian_suffix(a: int, b: int, c: int) -> str:
    return "x" * a + "y" * b + "z" * c


def _stored_var_names(l: int, prefix: str = "") -> List[str]:
    if l == 0:
        return [f"T{prefix}_{l}"]
    return [f"T{prefix}_{l}_{_cartesian_suffix(a, b, c)}" for (a, b, c) in stored_basis(l)]


def _grad_stored_var_names(l: int, prefix: str = "") -> List[str]:
    return [f"grad_{nm}" for nm in _stored_var_names(l, prefix)]


def _out_var_names(l_out: int) -> List[str]:
    if l_out == 0:
        return ["v"]
    return [_cartesian_suffix(a, b, c) for (a, b, c) in stored_basis(l_out)]


def _grad_out_var_names(l_out: int) -> List[str]:
    if l_out == 0:
        return ["grad_out"]
    return [f"grad_{_cartesian_suffix(a, b, c)}" for (a, b, c) in stored_basis(l_out)]


def _stored_to_triton_subs(l: int, input_idx: int, prefix: str = "") -> Dict[sp.Symbol, sp.Symbol]:
    if input_idx not in (1, 2):
        raise ValueError(f"input_idx must be 1 or 2, got {input_idx!r}.")
    if prefix not in ("", "1", "2"):
        raise ValueError(f"Bad prefix {prefix!r}; expected '', '1', or '2'.")
    syms = even_tp_t1_symbols(l) if input_idx == 1 else even_tp_t2_symbols(l)
    var_names = _stored_var_names(l, prefix)
    return {syms[i]: sp.Symbol(var_names[i], real=True) for i in range(2 * l + 1)}


def _emit_contract_function(l1: int, l2: int, l_out: int) -> str:
    if not even_tp_path_allowed(l1, l2, l_out):
        raise ValueError(f"Not an allowed even-TP path: ({l1}, {l2}, {l_out})")

    like_pair = l1 == l2
    p1, p2 = ("1", "2") if like_pair else ("", "")
    names_in1 = _stored_var_names(l1, p1)
    names_in2 = _stored_var_names(l2, p2)

    subs = {
        **_stored_to_triton_subs(l1, 1, p1),
        **_stored_to_triton_subs(l2, 2, p2),
    }
    exprs = even_tp_forward_sympy(l1, l2, l_out)

    out_names = _out_var_names(l_out)

    lines: List[str] = []
    lines.append("@triton.jit")
    args = ", ".join(names_in1 + names_in2)
    lines.append(f"def contract_l{l1}_l{l2}_to_l{l_out}({args}):")
    for name, expr in zip(out_names, exprs):
        rhs = _pycode(expr.xreplace(subs))
        lines.append(f"    {name} = {rhs}")
    if len(out_names) == 1:
        lines.append(f"    return {out_names[0]}")
    else:
        lines.append(f"    return {', '.join(out_names)}")
    return "\n".join(lines)


def _accum_grad_symbols(l1: int, l2: int, l_out: int):
    if not even_tp_path_allowed(l1, l2, l_out):
        raise ValueError(f"Not an allowed even-TP path: ({l1}, {l2}, {l_out})")

    like_pair = l1 == l2
    p1, p2 = ("1", "2") if like_pair else ("", "")
    names_in1 = _stored_var_names(l1, p1)
    names_in2 = _stored_var_names(l2, p2)
    subs = {
        **_stored_to_triton_subs(l1, 1, p1),
        **_stored_to_triton_subs(l2, 2, p2),
    }

    grad_out_names = _grad_out_var_names(l_out)
    g_syms = even_tp_g_symbols(l_out)
    g_subs = {g_syms[i]: sp.Symbol(grad_out_names[i], real=True) for i in range(2 * l_out + 1)}

    grad_t1_names = _grad_stored_var_names(l1, p1)
    grad_t2_names = _grad_stored_var_names(l2, p2)

    grad_T1_exprs, grad_T2_exprs = even_tp_backward_sympy(l1, l2, l_out)
    all_subs = {**subs, **g_subs}
    return (
        names_in1,
        names_in2,
        grad_out_names,
        grad_t1_names,
        grad_t2_names,
        grad_T1_exprs,
        grad_T2_exprs,
        all_subs,
    )


def _emit_accum_grad_def(
    fn_name: str,
    sig_args: List[str],
    name_expr_groups: Sequence[Tuple[List[str], List[sp.Expr]]],
    ret_items: List[str],
) -> str:
    lines: List[str] = []
    lines.append("@triton.jit")
    lines.append(f"def {fn_name}({', '.join(sig_args)}):")
    for names, exprs in name_expr_groups:
        for name, expr in zip(names, exprs):
            lines.append(f"    {name} = {name} + ({_pycode(expr)})")
    if len(ret_items) == 1:
        lines.append(f"    return {ret_items[0]}")
    else:
        lines.append(f"    return (")
        for nm in ret_items[:-1]:
            lines.append(f"        {nm},")
        lines.append(f"        {ret_items[-1]},")
        lines.append(f"    )")
    return "\n".join(lines)


def _emit_accum_grad_function(l1: int, l2: int, l_out: int) -> str:
    (
        names_in1,
        names_in2,
        grad_out_names,
        grad_t1_names,
        grad_t2_names,
        grad_T1_exprs,
        grad_T2_exprs,
        all_subs,
    ) = _accum_grad_symbols(l1, l2, l_out)
    grad_T1_subbed = [e.xreplace(all_subs) for e in grad_T1_exprs]
    grad_T2_subbed = [e.xreplace(all_subs) for e in grad_T2_exprs]
    return _emit_accum_grad_def(
        f"accum_grad_l{l1}_l{l2}_to_l{l_out}",
        grad_out_names + names_in1 + names_in2 + grad_t1_names + grad_t2_names,
        [(grad_t1_names, grad_T1_subbed), (grad_t2_names, grad_T2_subbed)],
        grad_t1_names + grad_t2_names,
    )


def _emit_accum_grad_function_one_side(l1: int, l2: int, l_out: int, side: str) -> str:
    if not even_tp_path_allowed(l1, l2, l_out):
        raise ValueError(f"Not an allowed even-TP path: ({l1}, {l2}, {l_out})")
    if side not in ("a", "b"):
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")

    (
        names_in1,
        names_in2,
        grad_out_names,
        grad_t1_names,
        grad_t2_names,
        grad_T1_exprs,
        grad_T2_exprs,
        all_subs,
    ) = _accum_grad_symbols(l1, l2, l_out)

    if side == "a":
        # Returns grad_T1; needs grad_out + T2.
        sig_args = grad_out_names + names_in2 + grad_t1_names
        names = grad_t1_names
        exprs = [e.xreplace(all_subs) for e in grad_T1_exprs]
    else:
        # Returns grad_T2; needs grad_out + T1.
        sig_args = grad_out_names + names_in1 + grad_t2_names
        names = grad_t2_names
        exprs = [e.xreplace(all_subs) for e in grad_T2_exprs]

    return _emit_accum_grad_def(
        f"accum_grad_l{l1}_l{l2}_to_l{l_out}_{side}",
        sig_args,
        [(names, exprs)],
        names,
    )


def _tp_path_order(kernel_l_max: int) -> List[Tuple[int, int, int]]:
    paths: List[Tuple[int, int, int]] = []
    for l_out in range(kernel_l_max + 1):
        for l1 in range(kernel_l_max + 1):
            for l2 in range(l1 + 1):  # l2 <= l1
                if even_tp_path_allowed(l1, l2, l_out):
                    paths.append((l1, l2, l_out))
    return paths


def _emit_path_ladder(
    kernel_l_max: int,
    emit_one_path: Callable[[int, int, int, bool, str], str],
    indent: str,
    lead_blank: bool = True,
    emit_path_zero: bool = True,
    swap_block: bool = True,
    emit_preamble: Optional[Callable[[int], str]] = None,
) -> str:
    paths = _tp_path_order(kernel_l_max)
    lead = "\n" if lead_blank else ""
    body: List[str] = []
    for l_out in range(kernel_l_max + 1):
        body.append(f"{lead}{indent}if OUT_L_MAX >= {l_out}:\n")
        if emit_path_zero:
            body.append(f"{indent}    path{l_out} = 0\n")
        if emit_preamble is not None:
            body.append(emit_preamble(l_out))
        l_out_paths = [p for p in paths if p[2] == l_out]
        for l1, l2, _lo in l_out_paths:
            body.append(emit_one_path(l1, l2, l_out, False, indent + "    "))
        swap_paths = [(l1, l2) for (l1, l2, _lo) in l_out_paths if l1 != l2]
        if swap_paths and swap_block:
            body.append(f"{indent}    if not SYMMETRIC_PRODUCT:\n")
        swap_indent = indent + ("        " if swap_block else "    ")
        for l1, l2 in swap_paths:
            body.append(emit_one_path(l1, l2, l_out, True, swap_indent))
    return "".join(body)


def _emit_load_l(l: int) -> str:
    var_names = _out_var_names(l)  # uses 'v' for l=0 -> we override below
    n = 2 * l + 1
    if l == 0:
        return (
            "@triton.jit\n"
            f"def load_l{l}(x_ptr, x_row, OFF: tl.constexpr, f_idx, mask, HAS_MASK: tl.constexpr, EVICT: tl.constexpr):\n"
            "    if HAS_MASK:\n"
            "        return tl.load(x_ptr + x_row + OFF + f_idx, mask=mask, other=0.0, eviction_policy=EVICT)\n"
            "    else:\n"
            "        return tl.load(x_ptr + x_row + OFF + f_idx, eviction_policy=EVICT)\n"
        )
    suffixes = [_cartesian_suffix(a, b, c) for (a, b, c) in stored_basis(l)]
    lines = ["@triton.jit"]
    lines.append(
        f"def load_l{l}(x_ptr, x_row, OFF: tl.constexpr, FEATURES: tl.constexpr, f_idx, mask, "
        "HAS_MASK: tl.constexpr, EVICT: tl.constexpr):"
    )
    lines.append("    base = x_ptr + x_row + OFF + f_idx")
    lines.append("    if HAS_MASK:")
    for i, suf in enumerate(suffixes):
        lines.append(f"        c_{suf} = tl.load(base + {i} * FEATURES, mask=mask, other=0.0, eviction_policy=EVICT)")
    lines.append("    else:")
    for i, suf in enumerate(suffixes):
        lines.append(f"        c_{suf} = tl.load(base + {i} * FEATURES, eviction_policy=EVICT)")
    lines.append(f"    return {', '.join('c_' + s for s in suffixes)}")
    return "\n".join(lines) + "\n"


def _emit_store_kernel(
    name: str,
    value_args: List[str],
    mask_arg: str,
    include_needs_scatter: bool,
    reduce_atomic: bool,
    use_base_vars: bool = True,
) -> str:
    lines: List[str] = []
    lines.append("@triton.jit")
    lines.append(f"def {name}(")
    lines.append("    out_ptr,")
    lines.append("    out_row,")
    lines.append("    OFFSET: tl.constexpr,")
    lines.append("    f_idx,")
    lines.append("    OUT_FEATURES: tl.constexpr,")
    lines.append("    N_PATHS: tl.constexpr,")
    lines.append("    PATH: tl.constexpr,")
    for a in value_args:
        lines.append(f"    {a},")
    lines.append(f"    {mask_arg},")
    if include_needs_scatter:
        lines.append("    NEEDS_SCATTER: tl.constexpr,")
    lines.append("    REDUCE_PATHS: tl.constexpr,")
    lines.append("):")

    def emit_stores(indent: str, base_expr: str, atomic: bool) -> None:
        for i, a in enumerate(value_args):
            addr = f"base + {i} * stride" if use_base_vars else base_expr
            if atomic:
                lines.append(f'{indent}tl.atomic_add({addr}, {a}, mask={mask_arg}, sem="relaxed")')
            else:
                lines.append(f"{indent}tl.store({addr}, {a}, mask={mask_arg})")

    reduce_base = "out_ptr + out_row + OFFSET + f_idx"
    path_base = "out_ptr + out_row + OFFSET + PATH * OUT_FEATURES + f_idx"
    lines.append("    if REDUCE_PATHS:")
    if use_base_vars:
        lines.append(f"        base = {reduce_base}")
        lines.append("        stride = OUT_FEATURES")
    emit_stores("        ", reduce_base, reduce_atomic)
    lines.append("    else:")
    if use_base_vars:
        lines.append(f"        base = {path_base}")
        lines.append("        stride = N_PATHS * OUT_FEATURES")
    if include_needs_scatter:
        lines.append("        if NEEDS_SCATTER:")
        emit_stores("            ", path_base, True)
        lines.append("        else:")
        emit_stores("            ", path_base, False)
    else:
        emit_stores("        ", path_base, False)
    return "\n".join(lines) + "\n"


def _emit_store_l_conditional(l: int) -> str:
    if l == 0:
        return _emit_store_kernel(
            "store_l0_conditional",
            ["val"],
            "mask",
            include_needs_scatter=True,
            reduce_atomic=True,
            use_base_vars=False,
        )
    suffixes = [_cartesian_suffix(a, b, c) for (a, b, c) in stored_basis(l)]
    return _emit_store_kernel(
        f"store_l{l}_conditional",
        suffixes,
        "mask",
        include_needs_scatter=True,
        reduce_atomic=True,
    )


def _emit_load_grad_l(l: int) -> str:
    if l == 0:
        return (
            "@triton.jit\n"
            "def load_grad_l0(\n"
            "    grad_out_ptr,\n"
            "    grad_out_row,\n"
            "    OFFSET: tl.constexpr,\n"
            "    f_idx,\n"
            "    GRAD_OUT_COL_STRIDE: tl.constexpr,\n"
            "    OUT_FEATURES: tl.constexpr,\n"
            "    PATH: tl.constexpr,\n"
            "    mask,\n"
            "    REDUCE_PATHS: tl.constexpr,\n"
            "):\n"
            "    if GRAD_OUT_COL_STRIDE == 0:\n"
            "        return tl.load(grad_out_ptr + grad_out_row, mask=mask, other=0.0)\n"
            "    col = (OFFSET + f_idx) * GRAD_OUT_COL_STRIDE\n"
            "    if REDUCE_PATHS:\n"
            "        return tl.load(grad_out_ptr + grad_out_row + col, mask=mask, other=0.0)\n"
            "    else:\n"
            "        return tl.load(\n"
            "            grad_out_ptr + grad_out_row + (OFFSET + PATH * OUT_FEATURES + f_idx) * GRAD_OUT_COL_STRIDE,\n"
            "            mask=mask, other=0.0,\n"
            "        )\n"
        )
    suffixes = [_cartesian_suffix(a, b, c) for (a, b, c) in stored_basis(l)]
    lines: List[str] = []
    lines.append("@triton.jit")
    lines.append(f"def load_grad_l{l}(")
    lines.append("    grad_out_ptr,")
    lines.append("    grad_out_row,")
    lines.append("    OFFSET: tl.constexpr,")
    lines.append("    f_idx,")
    lines.append("    GRAD_OUT_COL_STRIDE: tl.constexpr,")
    lines.append("    OUT_FEATURES: tl.constexpr,")
    lines.append("    N_PATHS: tl.constexpr,")
    lines.append("    PATH: tl.constexpr,")
    lines.append("    mask,")
    lines.append("    REDUCE_PATHS: tl.constexpr,")
    lines.append("):")
    lines.append("    if GRAD_OUT_COL_STRIDE == 0:")
    lines.append("        g = tl.load(grad_out_ptr + grad_out_row, mask=mask, other=0.0)")
    lines.append(f"        return {', '.join('g' for _ in suffixes)}")
    lines.append("    if REDUCE_PATHS:")
    lines.append("        stride = OUT_FEATURES")
    lines.append("        base = grad_out_ptr + grad_out_row + (OFFSET + f_idx) * GRAD_OUT_COL_STRIDE")
    lines.append("    else:")
    lines.append("        stride = N_PATHS * OUT_FEATURES")
    lines.append(
        "        base = grad_out_ptr + grad_out_row + (OFFSET + PATH * OUT_FEATURES + f_idx) * GRAD_OUT_COL_STRIDE"
    )
    for i, suf in enumerate(suffixes):
        lines.append(f"    g_{suf} = tl.load(base + {i} * stride * GRAD_OUT_COL_STRIDE, mask=mask, other=0.0)")
    lines.append(f"    return {', '.join('g_' + s for s in suffixes)}")
    return "\n".join(lines) + "\n"


def _emit_store_grad_l(l: int) -> str:
    if l == 0:
        return (
            "@triton.jit\n"
            "def store_grad_l0(\n"
            "    grad_ptr,\n"
            "    grad_row,\n"
            "    OFFSET: tl.constexpr,\n"
            "    f_idx,\n"
            "    b,\n"
            "    stride,\n"
            "    FEATURES: tl.constexpr,\n"
            "    grad_val,\n"
            "    f_mask,\n"
            "    b_mask,\n"
            "    mask,\n"
            "    NEEDS_SCATTER: tl.constexpr,\n"
            "):\n"
            "    if NEEDS_SCATTER:\n"
            "        if FEATURES == 1:\n"
            "            grad_masked = tl.where(f_mask, grad_val, 0.0)\n"
            "            grad_sum = tl.sum(grad_masked, axis=1)\n"
            "            grad_row_1d = tl.reshape(grad_row, [grad_row.shape[0]])\n"
            '            tl.atomic_add(grad_ptr + grad_row_1d + OFFSET, grad_sum, mask=b_mask, sem="relaxed")\n'
            "        else:\n"
            '            tl.atomic_add(grad_ptr + grad_row + OFFSET + f_idx, grad_val, mask=mask, sem="relaxed")\n'
            "    else:\n"
            "        if FEATURES == 1:\n"
            "            grad_masked = tl.where(f_mask, grad_val, 0.0)\n"
            "            grad_sum = tl.sum(grad_masked, axis=1)\n"
            "            grad_sum = tl.where(b_mask, grad_sum, 0.0)\n"
            '            tl.atomic_add(grad_ptr + b * stride + OFFSET, grad_sum, sem="relaxed")\n'
            "        else:\n"
            "            tl.store(grad_ptr + grad_row + OFFSET + f_idx, grad_val, mask=mask)\n"
        )
    suffixes = [_cartesian_suffix(a, b, c) for (a, b, c) in stored_basis(l)]
    arg_names = [f"grad_{s}" for s in suffixes]
    lines: List[str] = []
    lines.append("@triton.jit")
    lines.append(f"def store_grad_l{l}(")
    lines.append("    grad_ptr,")
    lines.append("    grad_row,")
    lines.append("    OFFSET: tl.constexpr,")
    lines.append("    f_idx,")
    lines.append("    b,")
    lines.append("    stride,")
    lines.append("    FEATURES: tl.constexpr,")
    for a in arg_names:
        lines.append(f"    {a},")
    lines.append("    f_mask,")
    lines.append("    b_mask,")
    lines.append("    mask,")
    lines.append("    NEEDS_SCATTER: tl.constexpr,")
    lines.append("):")
    lines.append("    if NEEDS_SCATTER:")
    lines.append("        if FEATURES == 1:")
    for a in arg_names:
        lines.append(f"            {a}_masked = tl.where(f_mask, {a}, 0.0)")
        lines.append(f"            {a}_sum = tl.sum({a}_masked, axis=1)")
    lines.append("            grad_row_1d = tl.reshape(grad_row, [grad_row.shape[0]])")
    lines.append("            base = grad_ptr + grad_row_1d + OFFSET")
    for i, a in enumerate(arg_names):
        lines.append(f'            tl.atomic_add(base + {i} * FEATURES, {a}_sum, mask=b_mask, sem="relaxed")')
    lines.append("        else:")
    lines.append("            base = grad_ptr + grad_row + OFFSET + f_idx")
    for i, a in enumerate(arg_names):
        lines.append(f'            tl.atomic_add(base + {i} * FEATURES, {a}, mask=mask, sem="relaxed")')
    lines.append("    else:")
    lines.append("        if FEATURES == 1:")
    for a in arg_names:
        lines.append(f"            {a}_masked = tl.where(f_mask, {a}, 0.0)")
        lines.append(f"            {a}_sum = tl.sum({a}_masked, axis=1)")
    lines.append("            base = grad_ptr + b * stride + OFFSET")
    for i, a in enumerate(arg_names):
        lines.append(f'            tl.atomic_add(base + {i} * FEATURES, {a}_sum, mask=b_mask, sem="relaxed")')
    lines.append("        else:")
    lines.append("            base = grad_ptr + grad_row + OFFSET + f_idx")
    for i, a in enumerate(arg_names):
        lines.append(f"            tl.store(base + {i} * FEATURES, {a}, mask=mask)")
    return "\n".join(lines) + "\n"


def _emit_accum_grad_weight_l(l: int) -> str:
    suffixes = [_cartesian_suffix(a, b, c) for (a, b, c) in stored_basis(l)] if l >= 1 else [""]
    if l == 0:
        grad_names = ["grad_out"]
        r_names = ["forward_unweighted"]
    else:
        grad_names = [f"grad_{s}" for s in suffixes]
        r_names = [f"r{s}" for s in suffixes]
    dot_expr = " + ".join(f"{g} * {r}" for g, r in zip(grad_names, r_names))
    lines: List[str] = []
    lines.append("@triton.jit")
    lines.append(f"def accum_grad_weight_l{l}(")
    lines.append("    grad_weights_ptr,")
    for a in grad_names + r_names:
        lines.append(f"    {a},")
    lines.append("    mask,")
    lines.append("    b_idx,")
    lines.append("    f_idx,")
    lines.append("    PATH_IDX: tl.constexpr,")
    lines.append("    N_TOTAL_PATHS: tl.constexpr,")
    lines.append("    OUT_FEATURES: tl.constexpr,")
    lines.append("    SHARED_WEIGHTS: tl.constexpr,")
    lines.append("):")
    lines.append("    f_mask = f_idx < OUT_FEATURES")
    lines.append("")
    lines.append("    if SHARED_WEIGHTS:")
    lines.append(f"        grad_w = tl.sum({dot_expr}, axis=0)")
    lines.append("        w_addr = grad_weights_ptr + PATH_IDX * OUT_FEATURES + f_idx")
    lines.append('        tl.atomic_add(w_addr, grad_w, mask=f_mask, sem="relaxed")')
    lines.append("    else:")
    lines.append(f"        grad_w_batch_feature = {dot_expr}")
    lines.append("        w_addr = (")
    lines.append("            grad_weights_ptr")
    lines.append("            + b_idx[:, None].to(tl.int64) * (N_TOTAL_PATHS * OUT_FEATURES)")
    lines.append("            + PATH_IDX * OUT_FEATURES")
    lines.append("            + f_idx[None, :]")
    lines.append("        )")
    lines.append("        tl.store(w_addr, grad_w_batch_feature, mask=mask)")
    return "\n".join(lines) + "\n"


_LOAD_WEIGHT_SRC = """\
@triton.jit
def load_weight(
    w_ptr,
    b_idx,
    f_idx,
    PATH_IDX: tl.constexpr,
    N_TOTAL_PATHS: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    SHARED_WEIGHTS: tl.constexpr,
    b_mask,
    f_mask,
):
    if SHARED_WEIGHTS:
        w_addr = w_ptr + PATH_IDX * OUT_FEATURES + f_idx
        w = tl.load(w_addr, mask=f_mask, other=1.0)
        return w[None, :]
    else:
        w_addr = (
            w_ptr
            + b_idx[:, None].to(tl.int64) * (N_TOTAL_PATHS * OUT_FEATURES)
            + PATH_IDX * OUT_FEATURES
            + f_idx[None, :]
        )
        mask = b_mask[:, None] & f_mask[None, :]
        return tl.load(w_addr, mask=mask, other=1.0)
"""


def _emit_low_level_helpers(kernel_l_max: int) -> str:
    parts: List[str] = []
    parts.append(_LOAD_WEIGHT_SRC + "\n")
    for l in range(kernel_l_max + 1):
        parts.append(_emit_load_l(l) + "\n")
    for l in range(kernel_l_max + 1):
        parts.append(_emit_store_l_conditional(l) + "\n")
    for l in range(kernel_l_max + 1):
        parts.append(_emit_load_grad_l(l) + "\n")
    for l in range(kernel_l_max + 1):
        parts.append(_emit_store_grad_l(l) + "\n")
    for l in range(kernel_l_max + 1):
        parts.append(_emit_accum_grad_weight_l(l) + "\n")
    return "".join(parts)


def _stored_input_var_list(l: int, side_prefix: str, name_prefix: str = "") -> List[str]:
    return [f"{name_prefix}{name}" for name in _stored_var_names(l, side_prefix)]


def _emit_load_input_slice(
    l: int,
    side_prefix: str,
    ptr: str,
    row_var: str,
    off_var: str,
    features_var: str,
    f_idx_var: str,
    mask_var: str,
    name_prefix: str = "",
    indent: str = "    ",
    evict: str = "",
    has_mask: str = "True",
) -> str:
    var_list = _stored_input_var_list(l, side_prefix, name_prefix)
    lhs = ", ".join(var_list)
    if l == 0:
        return (
            f"{indent}{lhs} = load_l0({ptr}, {row_var}, {off_var}, {f_idx_var}, {mask_var}, "
            f'{has_mask}, "{evict}")\n'
        )
    return (
        f"{indent}{lhs} = load_l{l}({ptr}, {row_var}, {off_var}, {features_var}, "
        f'{f_idx_var}, {mask_var}, {has_mask}, "{evict}")\n'
    )


def _emit_input_loads_block(
    side_prefix: str,
    ptr: str,
    row_var: str,
    off_var_prefix: str,
    features_var: str,
    f_idx_var: str,
    mask_var: str,
    l_max_var: str,
    kernel_l_max: int,
    name_prefix: str = "",
    indent: str = "        ",
    evict: str = "",
    has_mask: str = "True",
) -> str:
    lines: List[str] = []
    lines.append(
        _emit_load_input_slice(
            0,
            side_prefix,
            ptr,
            row_var,
            f"{off_var_prefix}_L0",
            features_var,
            f_idx_var,
            mask_var,
            name_prefix,
            indent,
            evict,
            has_mask,
        ).rstrip("\n")
    )
    for l in range(1, kernel_l_max + 1):
        lines.append(f"{indent}if {l_max_var} >= {l}:")
        lines.append(
            _emit_load_input_slice(
                l,
                side_prefix,
                ptr,
                row_var,
                f"{off_var_prefix}_L{l}",
                features_var,
                f_idx_var,
                mask_var,
                name_prefix,
                indent + "    ",
                evict,
                has_mask,
            ).rstrip("\n")
        )
    return "\n".join(lines) + "\n"


def _emit_grad_zeros_block(
    side_prefix: str,
    l_max_var: str,
    kernel_l_max: int,
    indent: str = "        ",
    base_var_prefix: str = "d_",
) -> str:
    lines: List[str] = []
    scalar_name = _stored_input_var_list(0, side_prefix)[0]
    lines.append(f"{indent}{base_var_prefix}{scalar_name} = tl.zeros_like({scalar_name})")
    for l in range(1, kernel_l_max + 1):
        lines.append(f"{indent}if {l_max_var} >= {l}:")
        var_list = _stored_input_var_list(l, side_prefix, "")
        d_var_list = [f"{base_var_prefix}{nm}" for nm in var_list]
        n = len(var_list)
        # Use the first stored component as a shape donor for tl.zeros_like.
        lines.append(
            f"{indent}    {', '.join(d_var_list)} = "
            f"{', '.join([f'tl.zeros_like({var_list[0]})' for _ in range(n)])}"
        )
    return "\n".join(lines) + "\n"


def _emit_store_grad_in_block(
    side_prefix: str,
    grad_ptr: str,
    grad_row_var: str,
    off_var_prefix: str,
    f_var: str,
    b_var: str,
    grad_stride: str,
    features_var: str,
    f_mask: str,
    b_mask: str,
    mask_var: str,
    needs_scatter_var: str,
    l_max_var: str,
    kernel_l_max: int,
    base_var_prefix: str = "d_",
    indent: str = "        ",
) -> str:
    lines: List[str] = []
    scalar_name = _stored_input_var_list(0, side_prefix)[0]
    lines.append(f"{indent}store_grad_l0(")
    lines.append(f"{indent}    {grad_ptr},")
    lines.append(f"{indent}    {grad_row_var},")
    lines.append(f"{indent}    {off_var_prefix}_L0,")
    lines.append(f"{indent}    {f_var}[None, :],")
    lines.append(f"{indent}    {b_var},")
    lines.append(f"{indent}    {grad_stride},")
    lines.append(f"{indent}    {features_var},")
    lines.append(f"{indent}    {base_var_prefix}{scalar_name},")
    lines.append(f"{indent}    {f_mask}[None, :],")
    lines.append(f"{indent}    {b_mask},")
    lines.append(f"{indent}    {mask_var},")
    lines.append(f"{indent}    {needs_scatter_var},")
    lines.append(f"{indent})")
    for l in range(1, kernel_l_max + 1):
        lines.append(f"{indent}if {l_max_var} >= {l}:")
        var_list = _stored_input_var_list(l, side_prefix, "")
        d_var_list = [f"{base_var_prefix}{nm}" for nm in var_list]
        lines.append(f"{indent}    store_grad_l{l}(")
        lines.append(f"{indent}        {grad_ptr},")
        lines.append(f"{indent}        {grad_row_var},")
        lines.append(f"{indent}        {off_var_prefix}_L{l},")
        lines.append(f"{indent}        {f_var}[None, :],")
        lines.append(f"{indent}        {b_var},")
        lines.append(f"{indent}        {grad_stride},")
        lines.append(f"{indent}        {features_var},")
        for dv in d_var_list:
            lines.append(f"{indent}        {dv},")
        lines.append(f"{indent}        {f_mask}[None, :],")
        lines.append(f"{indent}        {b_mask},")
        lines.append(f"{indent}        {mask_var},")
        lines.append(f"{indent}        {needs_scatter_var},")
        lines.append(f"{indent}    )")
    return "\n".join(lines) + "\n"


def _emit_out_off_constexpr_block(kernel_l_max: int, indent: str = "        ") -> str:
    lines = [f"{indent}OUT_L0_OFF = 0"]
    for l in range(1, kernel_l_max + 1):
        accumulands = [f"OUT_L{j}_SIZE" for j in range(l)]
        lines.append(f"{indent}OUT_L{l}_OFF = " + " + ".join(accumulands))
    return "\n".join(lines) + "\n"


def _path_inputs_for_kernel(l1: int, l2: int, l_out: int, swap: bool) -> Tuple[List[str], List[str]]:
    if not swap:
        side_a, side_b = "1", "2"
    else:
        side_a, side_b = "2", "1"
    return _stored_input_var_list(l1, side_a), _stored_input_var_list(l2, side_b)


_OUT_RESULT_VARS: Dict[int, List[str]] = {}


def _out_result_vars(l_out: int) -> List[str]:
    if l_out in _OUT_RESULT_VARS:
        return _OUT_RESULT_VARS[l_out]
    if l_out == 0:
        names = ["s0"]
    else:
        names = [f"r{_cartesian_suffix(a, b, c)}" for (a, b, c) in stored_basis(l_out)]
    _OUT_RESULT_VARS[l_out] = names
    return names


def _emit_fwd_path_branch(
    l1: int,
    l2: int,
    l_out: int,
    swap: bool,
    indent: str = "        ",
) -> str:
    if swap:
        guard = f"if IN1_L_MAX >= {l2} and IN2_L_MAX >= {l1}:"
    else:
        guard = f"if IN1_L_MAX >= {l1} and IN2_L_MAX >= {l2}:"
    args_a, args_b = _path_inputs_for_kernel(l1, l2, l_out, swap)
    args = args_a + args_b
    out_vars = _out_result_vars(l_out)
    lines: List[str] = []
    lines.append(f"{indent}{guard}")
    body = indent + "    "
    if len(out_vars) == 1:
        lines.append(f"{body}{out_vars[0]} = contract_l{l1}_l{l2}_to_l{l_out}(")
    else:
        lines.append(f"{body}{', '.join(out_vars)} = contract_l{l1}_l{l2}_to_l{l_out}(")
    for a in args[:-1]:
        lines.append(f"{body}    {a},")
    lines.append(f"{body}    {args[-1]},")
    lines.append(f"{body})")
    lines.append(f"{body}w = load_weight(")
    lines.append(f"{body}    weights_ptr,")
    lines.append(f"{body}    b,")
    lines.append(f"{body}    f,")
    lines.append(f"{body}    global_path_idx,")
    lines.append(f"{body}    N_TOTAL_PATHS,")
    lines.append(f"{body}    OUT_FEATURES,")
    lines.append(f"{body}    SHARED_WEIGHTS,")
    lines.append(f"{body}    b_mask,")
    lines.append(f"{body}    f_mask,")
    lines.append(f"{body})")
    # Apply weight component-wise.
    for o in out_vars:
        lines.append(f"{body}{o} = {o} * w")
    # Store output.
    lines.append(f"{body}store_l{l_out}_conditional(")
    lines.append(f"{body}    out_ptr,")
    lines.append(f"{body}    out_row,")
    lines.append(f"{body}    OUT_L{l_out}_OFF,")
    lines.append(f"{body}    f_idx,")
    lines.append(f"{body}    OUT_FEATURES,")
    lines.append(f"{body}    N_PATHS_L{l_out},")
    lines.append(f"{body}    path{l_out},")
    for o in out_vars:
        lines.append(f"{body}    {o},")
    lines.append(f"{body}    mo,")
    lines.append(f"{body}    NEEDS_SCATTER,")
    lines.append(f"{body}    REDUCE_PATHS,")
    lines.append(f"{body})")
    lines.append(f"{body}global_path_idx += 1")
    lines.append(f"{body}path{l_out} += 1")
    return "\n".join(lines) + "\n"


def _emit_kernel_setup_pid_block(indent: str = "    ") -> str:
    return (
        f"{indent}pid_b = tl.program_id(0)\n"
        f"{indent}pid_f = tl.program_id(1)\n"
        f"\n"
        f"{indent}b = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        f"{indent}b_mask = b < n_batch\n"
        f"\n"
        f"{indent}f = pid_f * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)\n"
        f"{indent}f_mask = f < OUT_FEATURES\n"
        f"\n"
        f"{indent}if IN1_FEATURES == 1:\n"
        f"{indent}    f1 = tl.zeros([FEATURE_BLOCK], dtype=tl.int32)\n"
        f"{indent}    f1_mask = tl.full([FEATURE_BLOCK], True, tl.int1)\n"
        f"{indent}else:\n"
        f"{indent}    f1 = f\n"
        f"{indent}    f1_mask = f_mask\n"
        f"\n"
        f"{indent}if IN2_FEATURES == 1:\n"
        f"{indent}    f2 = tl.zeros([FEATURE_BLOCK], dtype=tl.int32)\n"
        f"{indent}    f2_mask = tl.full([FEATURE_BLOCK], True, tl.int1)\n"
        f"{indent}else:\n"
        f"{indent}    f2 = f\n"
        f"{indent}    f2_mask = f_mask\n"
    )


def _emit_kernel_signature(
    name: str,
    pointers: List[str],
    int_args_before_features: List[str],
    int_args_after_features: List[str],
    kernel_l_max: int,
    include_out_l_sizes: bool,
    extra_constexpr_args: Optional[List[str]] = None,
    include_needs_scatter: bool = True,
    include_block_size: bool = True,
) -> str:
    lines: List[str] = []
    lines.append("@triton.jit")
    lines.append(f"def {name}(")
    for p in pointers:
        lines.append(f"    {p},")
    for a in int_args_before_features:
        lines.append(f"    {a},")
    lines.append("    IN1_FEATURES: tl.constexpr,")
    lines.append("    IN2_FEATURES: tl.constexpr,")
    lines.append("    OUT_FEATURES: tl.constexpr,")
    for a in int_args_after_features:
        lines.append(f"    {a},")
    lines.append("    IN1_L_MAX: tl.constexpr,")
    lines.append("    IN2_L_MAX: tl.constexpr,")
    lines.append("    OUT_L_MAX: tl.constexpr,")
    lines.append("    SYMMETRIC_PRODUCT: tl.constexpr,")
    lines.append("    SHARED_WEIGHTS: tl.constexpr,")
    if include_needs_scatter:
        lines.append("    NEEDS_SCATTER: tl.constexpr,")
    for l in range(kernel_l_max + 1):
        lines.append(f"    IN1_OFF_L{l}: tl.constexpr,")
    for l in range(kernel_l_max + 1):
        lines.append(f"    IN2_OFF_L{l}: tl.constexpr,")
    if include_out_l_sizes:
        for l in range(kernel_l_max + 1):
            lines.append(f"    OUT_L{l}_SIZE: tl.constexpr,")
    for l in range(kernel_l_max + 1):
        lines.append(f"    N_PATHS_L{l}: tl.constexpr,")
    lines.append("    N_TOTAL_PATHS: tl.constexpr,")
    lines.append("    REDUCE_PATHS: tl.constexpr,")
    for a in extra_constexpr_args or []:
        lines.append(f"    {a}: tl.constexpr,")
    if include_block_size:
        lines.append("    BLOCK_SIZE: tl.constexpr,")
    lines.append("    FEATURE_BLOCK: tl.constexpr,")
    lines.append("):")
    return "\n".join(lines) + "\n"


_TP_AUTOTUNE_HEADER = """\
_TP_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4, "FEATURE_BLOCK": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4, "FEATURE_BLOCK": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8, "FEATURE_BLOCK": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 128}, num_warps=4, num_stages=1),
]

_TP_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 128}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4, "FEATURE_BLOCK": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1, maxnreg=128),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1, maxnreg=96),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1, maxnreg=128),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 128}, num_warps=4, num_stages=1, maxnreg=128),
    triton.Config({"BLOCK_SIZE": 4, "FEATURE_BLOCK": 64}, num_warps=4, num_stages=1, maxnreg=96),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=1, num_stages=1),
]

_TP_DBWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4, "FEATURE_BLOCK": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 4, "FEATURE_BLOCK": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1, maxnreg=128),
    triton.Config({"BLOCK_SIZE": 2, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1, maxnreg=128),
    triton.Config({"BLOCK_SIZE": 1, "FEATURE_BLOCK": 64}, num_warps=2, num_stages=1, maxnreg=96),
]

_TP_FWD_AUTOTUNE_KEYS = [
    "IN1_FEATURES",
    "IN2_FEATURES",
    "OUT_FEATURES",
    "IN1_L_MAX",
    "IN2_L_MAX",
    "OUT_L_MAX",
    "SYMMETRIC_PRODUCT",
    "NEEDS_SCATTER",
]

_TP_BWD_AUTOTUNE_KEYS = _TP_FWD_AUTOTUNE_KEYS + [
    "NEED_GRAD_IN1",
    "NEED_GRAD_IN2",
    "NEED_GRAD_WEIGHTS",
    "PATH_GROUP",
]
_TP_DBWD_AUTOTUNE_KEYS = _TP_FWD_AUTOTUNE_KEYS + [
    "NEED_D_GRAD_OUT",
    "NEED_D_IN1",
    "NEED_D_IN2",
    "NEED_D_WEIGHTS",
    "HAS_V_WEIGHTS",
]

_TP_CSR_AUTOTUNE_CONFIGS = [
    triton.Config({"FEATURE_BLOCK": 32}, num_warps=1, num_stages=1),
    triton.Config({"FEATURE_BLOCK": 64}, num_warps=2, num_stages=1),
    triton.Config({"FEATURE_BLOCK": 128}, num_warps=4, num_stages=1),
]

_TP_BWD_CSR_CHUNK_NONE = 1 << 20

_TP_BWD_CSR_AUTOTUNE_CONFIGS = [
    triton.Config({"FEATURE_BLOCK": 32, "CHUNK": _TP_BWD_CSR_CHUNK_NONE}, num_warps=1, num_stages=1),
    triton.Config({"FEATURE_BLOCK": 32, "CHUNK": 32}, num_warps=1, num_stages=1),
    triton.Config({"FEATURE_BLOCK": 32, "CHUNK": 64}, num_warps=1, num_stages=1),
    triton.Config({"FEATURE_BLOCK": 32, "CHUNK": 32}, num_warps=1, num_stages=2),
    triton.Config({"FEATURE_BLOCK": 64, "CHUNK": 32}, num_warps=2, num_stages=1),
    triton.Config({"FEATURE_BLOCK": 128, "CHUNK": 32}, num_warps=4, num_stages=1),
]

_TP_FWD_CSR_AUTOTUNE_KEYS = [
    "IN1_FEATURES",
    "IN2_FEATURES",
    "OUT_FEATURES",
    "IN1_L_MAX",
    "IN2_L_MAX",
    "OUT_L_MAX",
    "SHARED_WEIGHTS",
    "REDUCE_PATHS",
    "PATH_GROUP",
]

if KERNEL_L_MAX >= 4:
    # hasattr guard: the codegen tests exec this module with a stubbed triton
    # whose Config() returns None; under the stub the filter must be a no-op.
    _TP_BWD_AUTOTUNE_CONFIGS = [
        c
        for c in _TP_BWD_AUTOTUNE_CONFIGS
        if not (
            hasattr(c, "kwargs")
            and c.kwargs["BLOCK_SIZE"] >= 4
            and c.num_warps >= 4
            and getattr(c, "maxnreg", None) is not None
        )
    ]

"""


def _emit_tp_fwd_kernel(kernel_l_max: int) -> str:
    sig = _emit_kernel_signature(
        "tp_fwd_kernel",
        pointers=[
            "in1_ptr",
            "in2_ptr",
            "out_ptr",
            "weights_ptr",
            "idx_i_ptr",
            "idx_j_ptr",
        ],
        int_args_before_features=[
            "n_batch",
        ],
        int_args_after_features=[
            "in1_stride",
            "in2_stride",
            "out_stride",
        ],
        kernel_l_max=kernel_l_max,
        include_out_l_sizes=True,
        include_needs_scatter=True,
    )
    ind = "    "
    body: List[str] = []
    body.append(_emit_kernel_setup_pid_block(ind))
    body.append(f"\n{ind}m2 = b_mask[:, None] & f2_mask[None, :]\n\n")
    # Scatter is served by the CSR kernel by default; the edge-parallel
    # atomic-scatter path here is the switchable fallback
    # (FLASHCART_TP_SCATTER_KERNEL=edge), kept for large-L exploration.
    body.append(f"{ind}m1 = b_mask[:, None] & f1_mask[None, :]\n")
    body.append(f"{ind}mo = b_mask[:, None] & f_mask[None, :]\n")
    body.append(f"{ind}if NEEDS_SCATTER:\n")
    body.append(f"{ind}    idx_j = tl.load(idx_j_ptr + b, mask=b_mask, other=0)\n")
    body.append(f"{ind}    idx_i = tl.load(idx_i_ptr + b, mask=b_mask, other=0)\n")
    body.append(f"{ind}    in1_row = idx_j[:, None].to(tl.int64) * in1_stride\n")
    body.append(f"{ind}    out_row = idx_i[:, None].to(tl.int64) * out_stride\n")
    body.append(f"{ind}else:\n")
    body.append(f"{ind}    in1_row = b[:, None].to(tl.int64) * in1_stride\n")
    body.append(f"{ind}    out_row = b[:, None].to(tl.int64) * out_stride\n")
    body.append(f"{ind}in2_row = b[:, None].to(tl.int64) * in2_stride\n\n")
    body.append(f"{ind}f_idx = f[None, :]\n")
    body.append(f"{ind}f1_idx = f1[None, :]\n")
    body.append(f"{ind}f2_idx = f2[None, :]\n\n")
    body.append(
        _emit_input_loads_block(
            "1",
            "in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "m1",
            "IN1_L_MAX",
            kernel_l_max,
            name_prefix="",
            indent=ind,
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "m2",
            "IN2_L_MAX",
            kernel_l_max,
            name_prefix="",
            indent=ind,
        )
    )
    body.append("\n")
    body.append(_emit_out_off_constexpr_block(kernel_l_max, indent=ind))
    body.append(f"\n{ind}global_path_idx = 0\n")
    body.append(_emit_path_ladder(kernel_l_max, _emit_fwd_path_branch, ind))
    return sig + "".join(body)


def _path_grad_inputs(l1: int, l2: int, swap: bool) -> Tuple[List[str], List[str], List[str], List[str]]:
    side_a, side_b = ("1", "2") if not swap else ("2", "1")
    inputs_a = _stored_input_var_list(l1, side_a)
    inputs_b = _stored_input_var_list(l2, side_b)
    grads_a = [f"d_{nm}" for nm in inputs_a]
    grads_b = [f"d_{nm}" for nm in inputs_b]
    return inputs_a, inputs_b, grads_a, grads_b


def _grad_out_load_line(l_out: int, lhs: str, indent: str) -> str:
    if l_out == 0:
        return (
            f"{indent}{lhs} = load_grad_l0(grad_out_ptr, grad_out_row, "
            f"OUT_L0_OFF, f_idx, GRAD_OUT_COL_STRIDE, OUT_FEATURES, path0, mo, REDUCE_PATHS)"
        )
    return (
        f"{indent}{lhs} = load_grad_l{l_out}("
        f"grad_out_ptr, grad_out_row, OUT_L{l_out}_OFF, f_idx, GRAD_OUT_COL_STRIDE, "
        f"OUT_FEATURES, N_PATHS_L{l_out}, path{l_out}, mo, REDUCE_PATHS)"
    )


def _emit_bwd_preload_grad_l_out(l_out: int, indent: str, group: bool = False) -> str:
    grad_out_lhs = ", ".join(_grad_out_var_names(l_out))
    ind = indent
    lines: List[str] = []
    if group:
        lines.append(f"{ind}if REDUCE_PATHS and GROUP_L{l_out} == PATH_GROUP:\n")
    else:
        lines.append(f"{ind}if REDUCE_PATHS:\n")
    lines.append(_grad_out_load_line(l_out, grad_out_lhs, ind + "    "))
    return "\n".join(lines) + "\n"


def _emit_bwd_path_branch(
    l1: int,
    l2: int,
    l_out: int,
    swap: bool,
    indent: str = "    ",
    group: bool = False,
    preloaded: bool = False,
) -> str:
    if swap:
        guard = f"if IN1_L_MAX >= {l2} and IN2_L_MAX >= {l1}:"
    else:
        guard = f"if IN1_L_MAX >= {l1} and IN2_L_MAX >= {l2}:"
    inputs_a, inputs_b, grads_a, grads_b = _path_grad_inputs(l1, l2, swap)
    out_vars = _out_result_vars(l_out)
    grad_out_names = _grad_out_var_names(l_out)
    lines: List[str] = []
    lines.append(f"{indent}{guard}")
    outer = indent + "    "
    body = outer + ("    " if group else "")
    if group:
        lines.append(f"{outer}if GROUP_L{l_out} == PATH_GROUP:")
    grad_out_lhs = ", ".join(grad_out_names)
    if preloaded:
        go_names = _csr_acc_names(l1, l2, l_out, swap, prefix="go")
        for gn, on in zip(grad_out_names, go_names):
            lines.append(f"{body}{gn} = {on}")
    else:
        lines.append(f"{body}if not REDUCE_PATHS:\n")
        lines.append(_grad_out_load_line(l_out, grad_out_lhs, body + "    "))
    args = inputs_a + inputs_b
    lines.append(f"{body}if NEED_GRAD_WEIGHTS:")
    weight_body = body + "    "
    if len(out_vars) == 1:
        lines.append(f"{weight_body}{out_vars[0]} = contract_l{l1}_l{l2}_to_l{l_out}(")
    else:
        lines.append(f"{weight_body}{', '.join(out_vars)} = contract_l{l1}_l{l2}_to_l{l_out}(")
    for a in args[:-1]:
        lines.append(f"{weight_body}    {a},")
    lines.append(f"{weight_body}    {args[-1]},")
    lines.append(f"{weight_body})")
    lines.append(f"{weight_body}accum_grad_weight_l{l_out}(")
    lines.append(f"{weight_body}    grad_weights_ptr,")
    for nm in grad_out_names:
        lines.append(f"{weight_body}    {nm},")
    for nm in out_vars:
        lines.append(f"{weight_body}    {nm},")
    lines.append(f"{weight_body}    mo, b, f, global_path_idx, N_TOTAL_PATHS, OUT_FEATURES, SHARED_WEIGHTS,")
    lines.append(f"{weight_body})")
    lines.append(f"{body}if NEED_GRAD_IN1 or NEED_GRAD_IN2:")
    input_body = body + "    "
    lines.append(f"{input_body}w = load_weight(")
    lines.append(f"{input_body}    weights_ptr, b, f, global_path_idx, N_TOTAL_PATHS,")
    lines.append(f"{input_body}    OUT_FEATURES, SHARED_WEIGHTS, b_mask, f_mask,")
    lines.append(f"{input_body})")
    weighted_grad_names = []
    for nm in grad_out_names:
        wnm = f"{nm}_w"
        weighted_grad_names.append(wnm)
        lines.append(f"{input_body}{wnm} = {nm} * w")
    # grad_in1
    if not swap:
        in1_helper, in1_lhs, in1_inputs = "a", grads_a, inputs_b
    else:
        in1_helper, in1_lhs, in1_inputs = "b", grads_b, inputs_a
    lines.append(f"{body}if NEED_GRAD_IN1:")
    input_body = body + "    "
    lines.append(f"{input_body}{', '.join(in1_lhs)} = accum_grad_l{l1}_l{l2}_to_l{l_out}_{in1_helper}(")
    for nm in weighted_grad_names:
        lines.append(f"{input_body}    {nm},")
    for nm in in1_inputs:
        lines.append(f"{input_body}    {nm},")
    for nm in in1_lhs[:-1]:
        lines.append(f"{input_body}    {nm},")
    lines.append(f"{input_body}    {in1_lhs[-1]},")
    lines.append(f"{input_body})")
    # grad_in2
    if not swap:
        in2_helper, in2_lhs, in2_inputs = "b", grads_b, inputs_a
    else:
        in2_helper, in2_lhs, in2_inputs = "a", grads_a, inputs_b
    lines.append(f"{body}if NEED_GRAD_IN2:")
    input_body = body + "    "
    lines.append(f"{input_body}{', '.join(in2_lhs)} = accum_grad_l{l1}_l{l2}_to_l{l_out}_{in2_helper}(")
    for nm in weighted_grad_names:
        lines.append(f"{input_body}    {nm},")
    for nm in in2_inputs:
        lines.append(f"{input_body}    {nm},")
    for nm in in2_lhs[:-1]:
        lines.append(f"{input_body}    {nm},")
    lines.append(f"{input_body}    {in2_lhs[-1]},")
    lines.append(f"{input_body})")
    lines.append(f"{outer}global_path_idx += 1")
    lines.append(f"{outer}path{l_out} += 1")
    return "\n".join(lines) + "\n"


def _emit_tp_bwd_kernel(kernel_l_max: int) -> str:
    pointers = [
        "grad_out_ptr",
        "in1_ptr",
        "in2_ptr",
        "weights_ptr",
        "grad_in1_ptr",
        "grad_in2_ptr",
        "grad_weights_ptr",
        "idx_i_ptr",
        "idx_j_ptr",
    ]
    sig = _emit_kernel_signature(
        "tp_bwd_kernel",
        pointers=pointers,
        int_args_before_features=["n_batch"],
        int_args_after_features=[
            "in1_stride",
            "in2_stride",
            "grad_in1_stride",
            "grad_in2_stride",
        ],
        kernel_l_max=kernel_l_max,
        include_out_l_sizes=True,
        extra_constexpr_args=[
            "GRAD_OUT_STRIDE",
            "GRAD_OUT_COL_STRIDE",
            "NEED_GRAD_IN1",
            "NEED_GRAD_IN2",
            "NEED_GRAD_WEIGHTS",
            "PATH_GROUP",
            "N_PATH_GROUPS",
            *[f"GROUP_L{l}" for l in range(kernel_l_max + 1)],
        ],
    )
    ind = "    "
    body: List[str] = []
    body.append(_emit_kernel_setup_pid_block(ind))
    body.append(f"\n{ind}m1 = b_mask[:, None] & f1_mask[None, :]\n")
    body.append(f"\n{ind}m2 = b_mask[:, None] & f2_mask[None, :]\n")
    body.append(f"{ind}mo = b_mask[:, None] & f_mask[None, :]\n\n")
    body.append(f"{ind}if NEEDS_SCATTER:\n")
    body.append(f"{ind}    idx_j = tl.load(idx_j_ptr + b, mask=b_mask, other=0)\n")
    body.append(f"{ind}    idx_i = tl.load(idx_i_ptr + b, mask=b_mask, other=0)\n")
    body.append(f"{ind}    in1_row = idx_j[:, None].to(tl.int64) * in1_stride\n")
    body.append(f"{ind}    in2_row = b[:, None].to(tl.int64) * in2_stride\n")
    body.append(f"{ind}    grad_out_row = idx_i[:, None].to(tl.int64) * GRAD_OUT_STRIDE\n")
    body.append(f"{ind}    grad_in1_row = idx_j[:, None].to(tl.int64) * grad_in1_stride\n")
    body.append(f"{ind}    grad_in2_row = b[:, None].to(tl.int64) * grad_in2_stride\n")
    body.append(f"{ind}else:\n")
    body.append(f"{ind}    in1_row = b[:, None].to(tl.int64) * in1_stride\n")
    body.append(f"{ind}    in2_row = b[:, None].to(tl.int64) * in2_stride\n")
    body.append(f"{ind}    grad_out_row = b[:, None].to(tl.int64) * GRAD_OUT_STRIDE\n")
    body.append(f"{ind}    grad_in1_row = b[:, None].to(tl.int64) * grad_in1_stride\n")
    body.append(f"{ind}    grad_in2_row = b[:, None].to(tl.int64) * grad_in2_stride\n")
    body.append("\n")
    body.append(f"{ind}f_idx = f[None, :]\n")
    body.append(f"{ind}f1_idx = f1[None, :]\n")
    body.append(f"{ind}f2_idx = f2[None, :]\n\n")
    body.append(
        _emit_input_loads_block(
            "1",
            "in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "m1",
            "IN1_L_MAX",
            kernel_l_max,
            indent=ind,
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "m2",
            "IN2_L_MAX",
            kernel_l_max,
            indent=ind,
        )
    )
    body.append("\n")
    body.append(f"{ind}if NEED_GRAD_IN1:\n")
    body.append(_emit_grad_zeros_block("1", "IN1_L_MAX", kernel_l_max, indent=ind + "    "))
    body.append(f"{ind}if NEED_GRAD_IN2:\n")
    body.append(_emit_grad_zeros_block("2", "IN2_L_MAX", kernel_l_max, indent=ind + "    "))
    body.append("\n")
    body.append(_emit_out_off_constexpr_block(kernel_l_max, indent=ind))
    body.append(f"\n{ind}global_path_idx = 0\n")
    body.append(
        _emit_path_ladder(
            kernel_l_max,
            lambda l1, l2, l_out, swap, indent: _emit_bwd_path_branch(
                l1, l2, l_out, swap=swap, indent=indent, group=True
            ),
            ind,
            emit_preamble=lambda l_out: _emit_bwd_preload_grad_l_out(l_out, indent=ind + "    ", group=True),
        )
    )
    body.append("\n")
    body.append(f"{ind}if NEED_GRAD_IN1:\n")
    body.append(
        _emit_store_grad_in_block(
            "1",
            "grad_in1_ptr",
            "grad_in1_row",
            "IN1_OFF",
            "f1",
            "b",
            "grad_in1_stride",
            "IN1_FEATURES",
            "f_mask",
            "b_mask",
            "m1",
            "NEEDS_SCATTER",
            "IN1_L_MAX",
            kernel_l_max,
            indent=ind + "    ",
        )
    )
    body.append("\n")
    body.append(f"{ind}if NEED_GRAD_IN2:\n")
    body.append(
        _emit_store_grad_in_block(
            "2",
            "grad_in2_ptr",
            "grad_in2_row",
            "IN2_OFF",
            "f2",
            "b",
            "grad_in2_stride",
            "IN2_FEATURES",
            "f_mask",
            "b_mask",
            "m2",
            "N_PATH_GROUPS > 1",
            "IN2_L_MAX",
            kernel_l_max,
            indent=ind + "    ",
        )
    )
    return sig + "".join(body)


def _emit_bwd_csr_grad_out_preload(kernel_l_max: int, indent: str) -> str:
    def one_path(l1: int, l2: int, l_out: int, swap: bool, ind: str) -> str:
        guard = _csr_path_guard(l1, l2, swap)
        if swap:
            guard = f"not SYMMETRIC_PRODUCT and {guard}"
        go_lhs = ", ".join(_csr_acc_names(l1, l2, l_out, swap, prefix="go"))
        lines = [
            f"{ind}if {guard}:",
            f"{ind}    if GROUP_L{l_out} == PATH_GROUP:",
            _grad_out_load_line(l_out, go_lhs, ind + "        "),
            f"{ind}    path{l_out} += 1",
        ]
        return "\n".join(lines) + "\n"

    return _emit_path_ladder(
        kernel_l_max,
        one_path,
        indent,
        lead_blank=False,
        swap_block=False,
    )


def _emit_tp_bwd_csr_kernel(kernel_l_max: int) -> str:
    sig = _emit_kernel_signature(
        "tp_bwd_csr_kernel",
        pointers=[
            "grad_out_ptr",
            "in1_ptr",
            "in2_ptr",
            "weights_ptr",
            "grad_in1_ptr",
            "grad_in2_ptr",
            "grad_weights_ptr",
            "idx_j_ptr",
            "rowptr_ptr",
        ],
        int_args_before_features=[
            "in1_stride",
            "in2_stride",
            "grad_in1_stride",
            "grad_in2_stride",
        ],
        int_args_after_features=[],
        kernel_l_max=kernel_l_max,
        include_out_l_sizes=True,
        extra_constexpr_args=[
            "GRAD_OUT_STRIDE",
            "GRAD_OUT_COL_STRIDE",
            "NEED_GRAD_IN1",
            "NEED_GRAD_IN2",
            "NEED_GRAD_WEIGHTS",
            "PATH_GROUP",
            "N_PATH_GROUPS",
            *[f"GROUP_L{l}" for l in range(kernel_l_max + 1)],
            "N_CHUNK_SLOTS",
            "CHUNK",
        ],
        include_needs_scatter=False,
        include_block_size=False,
    )

    ind = "    "
    body: List[str] = []
    # BS=1 setup: reuse the 2-D helpers with a length-1 batch axis.
    body.append(
        _emit_csr_setup_block(
            [
                "pid0 = tl.program_id(0)",
                "pid_f = tl.program_id(1)",
                "pid_n = pid0 // N_CHUNK_SLOTS",
                "pid_c = pid0 % N_CHUNK_SLOTS",
            ],
            ind,
            two_d=True,
        )
    )
    body.append(_emit_out_off_constexpr_block(kernel_l_max, indent=ind))
    body.append(f"{ind}row_start = tl.load(rowptr_ptr + pid_n).to(tl.int64)\n")
    body.append(f"{ind}row_end = tl.load(rowptr_ptr + pid_n + 1).to(tl.int64)\n")
    body.append(f"{ind}chunk_first = row_start + pid_c * CHUNK\n")
    body.append(f"{ind}if chunk_first < row_end:\n")
    pre = ind + "    "
    body.append(f"{pre}grad_out_row = (pid_n.to(tl.int64) * GRAD_OUT_STRIDE) + tl.zeros([1, 1], tl.int64)\n")
    body.append(_emit_bwd_csr_grad_out_preload(kernel_l_max, pre))
    body.append("\n")
    body.append(f"{pre}for chunk_start in range(chunk_first, row_end, N_CHUNK_SLOTS * CHUNK):\n")
    body.append(f"{pre}    chunk_end = tl.minimum(chunk_start + CHUNK, row_end)\n")
    body.append(f"{pre}    for edge in range(chunk_start, chunk_end):\n")
    loop = pre + "        "
    body.append(f"{loop}j = tl.load(idx_j_ptr + edge)\n")
    body.append(f"{loop}b = edge + tl.zeros([1], tl.int32)\n")
    body.append(f"{loop}in1_row = j.to(tl.int64) * in1_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(f"{loop}in2_row = edge.to(tl.int64) * in2_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(f"{loop}grad_in1_row = j.to(tl.int64) * grad_in1_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(f"{loop}grad_in2_row = edge.to(tl.int64) * grad_in2_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(
        _emit_input_loads_block(
            "1",
            "in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "m1",
            "IN1_L_MAX",
            kernel_l_max,
            indent=loop,
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "m2",
            "IN2_L_MAX",
            kernel_l_max,
            indent=loop,
        )
    )
    body.append(f"{loop}if NEED_GRAD_IN1:\n")
    body.append(_emit_grad_zeros_block("1", "IN1_L_MAX", kernel_l_max, indent=loop + "    "))
    body.append(f"{loop}if NEED_GRAD_IN2:\n")
    body.append(_emit_grad_zeros_block("2", "IN2_L_MAX", kernel_l_max, indent=loop + "    "))
    body.append(f"{loop}global_path_idx = 0\n")
    body.append(
        _emit_path_ladder(
            kernel_l_max,
            lambda l1, l2, l_out, swap, indent: _emit_bwd_path_branch(
                l1, l2, l_out, swap=swap, indent=indent, group=True, preloaded=True
            ),
            loop,
            lead_blank=False,
        )
    )
    body.append(f"{loop}if NEED_GRAD_IN1:\n")
    body.append(
        _emit_store_grad_in_block(
            "1",
            "grad_in1_ptr",
            "grad_in1_row",
            "IN1_OFF",
            "f1",
            "b",
            "grad_in1_stride",
            "IN1_FEATURES",
            "f_mask",
            "b_mask",
            "m1",
            "True",
            "IN1_L_MAX",
            kernel_l_max,
            indent=loop + "    ",
        )
    )
    body.append(f"{loop}if NEED_GRAD_IN2:\n")
    body.append(
        _emit_store_grad_in_block(
            "2",
            "grad_in2_ptr",
            "grad_in2_row",
            "IN2_OFF",
            "f2",
            "b",
            "grad_in2_stride",
            "IN2_FEATURES",
            "f_mask",
            "b_mask",
            "m2",
            "N_PATH_GROUPS > 1",
            "IN2_L_MAX",
            kernel_l_max,
            indent=loop + "    ",
        )
    )
    return sig + "".join(body)


def _emit_dbwd_path_branch(
    l1: int,
    l2: int,
    l_out: int,
    swap: bool,
    indent: str = "    ",
    group: bool = False,
    preloaded: bool = False,
    accumulate_d_grad_out: bool = False,
) -> str:
    if swap:
        guard = f"if IN1_L_MAX >= {l2} and IN2_L_MAX >= {l1}:"
    else:
        guard = f"if IN1_L_MAX >= {l1} and IN2_L_MAX >= {l2}:"
    side_a, side_b = ("1", "2") if not swap else ("2", "1")
    T_a = _stored_input_var_list(l1, side_a)
    T_b = _stored_input_var_list(l2, side_b)
    vT_a = _stored_input_var_list(l1, side_a, "v")
    vT_b = _stored_input_var_list(l2, side_b, "v")
    d_T_a = [f"d_{nm}" for nm in T_a]
    d_T_b = [f"d_{nm}" for nm in T_b]

    out_vars = _out_result_vars(l_out)
    u_c_names = [f"uc_{nm}" for nm in out_vars]
    u_ab_names = [f"uab_{nm}" for nm in out_vars]
    u_ab2_names = [f"uab2_{nm}" for nm in out_vars]
    eff_T_a = [f"eff_{nm}" for nm in T_a]
    eff_T_b = [f"eff_{nm}" for nm in T_b]
    d_grad_out_names = (
        [f"d_grad_{_cartesian_suffix(*abc)}" for abc in stored_basis(l_out)] if l_out >= 1 else ["d_grad_out_p"]
    )
    grad_out_names = _grad_out_var_names(l_out)

    lines: List[str] = []
    lines.append(f"{indent}{guard}")
    outer = indent + "    "
    body = outer + ("    " if group else "")
    if group:
        lines.append(f"{outer}if GROUP_L{l_out} == PATH_GROUP:")

    def emit_u_ab(line_body: str = None) -> None:
        lb = body if line_body is None else line_body
        lines.append(f"{lb}{', '.join(u_ab_names)} = contract_l{l1}_l{l2}_to_l{l_out}(")
        for a in vT_a + T_b:
            lines.append(f"{lb}    {a},")
        lines.append(f"{lb})")
        lines.append(f"{lb}{', '.join(u_ab2_names)} = contract_l{l1}_l{l2}_to_l{l_out}(")
        for a in T_a + vT_b:
            lines.append(f"{lb}    {a},")
        lines.append(f"{lb})")
        for u, u2 in zip(u_ab_names, u_ab2_names):
            lines.append(f"{lb}{u} = {u} + {u2}")

    def emit_load_grad_out(line_body: str = body) -> None:
        grad_out_lhs = ", ".join(grad_out_names)
        if l_out == 0:
            lines.append(
                f"{line_body}{grad_out_lhs} = load_grad_l0(grad_out_ptr, grad_out_row, "
                f"OUT_L0_OFF, f_idx, GRAD_OUT_COL_STRIDE, OUT_FEATURES, path0, mo, REDUCE_PATHS)"
            )
        else:
            lines.append(
                f"{line_body}{grad_out_lhs} = load_grad_l{l_out}("
                f"grad_out_ptr, grad_out_row, OUT_L{l_out}_OFF, f_idx, GRAD_OUT_COL_STRIDE, "
                f"OUT_FEATURES, N_PATHS_L{l_out}, path{l_out}, mo, REDUCE_PATHS)"
            )

    def emit_load_weights(load_vw: bool) -> None:
        lines.append(
            f"{body}w = load_weight(weights_ptr, b, f, global_path_idx, N_TOTAL_PATHS, "
            f"OUT_FEATURES, SHARED_WEIGHTS, b_mask, f_mask)"
        )
        if load_vw:
            lines.append(f"{body}if HAS_V_WEIGHTS:")
            lines.append(
                f"{body}    vw = load_weight(v_weights_ptr, b, f, global_path_idx, N_TOTAL_PATHS, "
                f"OUT_FEATURES, SHARED_WEIGHTS, b_mask, f_mask)"
            )

    if preloaded:
        go_names = _csr_acc_names(l1, l2, l_out, swap, prefix="go")
        lines.append(f"{body}if NEED_D_IN1 or NEED_D_IN2 or NEED_D_WEIGHTS:")
        for gn, on in zip(grad_out_names, go_names):
            lines.append(f"{body}    {gn} = {on}")
    else:
        lines.append(f"{body}if not REDUCE_PATHS:\n")
        emit_load_grad_out(body + "    ")
    lines.append(f"{body}if NEED_D_GRAD_OUT and HAS_V_WEIGHTS:")
    weight_body = body + "    "
    lines.append(f"{weight_body}{', '.join(u_c_names)} = contract_l{l1}_l{l2}_to_l{l_out}(")
    for a in T_a + T_b:
        lines.append(f"{weight_body}    {a},")
    lines.append(f"{weight_body})")
    lines.append(f"{body}if NEED_D_GRAD_OUT or NEED_D_WEIGHTS:")
    emit_u_ab(body + "    ")
    emit_load_weights(load_vw=True)
    lines.append(f"{body}if NEED_D_GRAD_OUT:")
    dgo_body = body + "    "
    for d, u, uc in zip(d_grad_out_names, u_ab_names, u_c_names):
        lines.append(f"{dgo_body}if HAS_V_WEIGHTS:")
        lines.append(f"{dgo_body}    {d} = w * {u} + vw * {uc}")
        lines.append(f"{dgo_body}else:")
        lines.append(f"{dgo_body}    {d} = w * {u}")
    if accumulate_d_grad_out:
        dgo_acc = _csr_acc_names(l1, l2, l_out, swap, prefix="dgo")
        dgo_red = _csr_reduce_acc_names(l_out, prefix="dgo")
        lines.append(f"{dgo_body}if REDUCE_PATHS:")
        for rn, d in zip(dgo_red, d_grad_out_names):
            lines.append(f"{dgo_body}    {rn} = {rn} + {d}")
        lines.append(f"{dgo_body}else:")
        for an, d in zip(dgo_acc, d_grad_out_names):
            lines.append(f"{dgo_body}    {an} = {an} + {d}")
    else:
        lines.append(f"{dgo_body}store_l{l_out}_conditional(")
        lines.append(f"{dgo_body}    d_grad_out_ptr,")
        lines.append(f"{dgo_body}    d_grad_out_row,")
        lines.append(f"{dgo_body}    OUT_L{l_out}_OFF,")
        lines.append(f"{dgo_body}    f_idx,")
        lines.append(f"{dgo_body}    OUT_FEATURES,")
        lines.append(f"{dgo_body}    N_PATHS_L{l_out},")
        lines.append(f"{dgo_body}    path{l_out},")
        for d in d_grad_out_names:
            lines.append(f"{dgo_body}    {d},")
        lines.append(f"{dgo_body}    mo,")
        lines.append(f"{dgo_body}    NEEDS_SCATTER,")
        lines.append(f"{dgo_body}    REDUCE_PATHS,")
        lines.append(f"{dgo_body})")
    lines.append(f"{body}if NEED_D_WEIGHTS:")
    dw_body = body + "    "
    lines.append(f"{dw_body}accum_grad_weight_l{l_out}(")
    lines.append(f"{dw_body}    d_weights_ptr,")
    for nm in grad_out_names:
        lines.append(f"{dw_body}    {nm},")
    for nm in u_ab_names:
        lines.append(f"{dw_body}    {nm},")
    lines.append(f"{dw_body}    mo, b, f, global_path_idx, N_TOTAL_PATHS, OUT_FEATURES, SHARED_WEIGHTS,")
    lines.append(f"{dw_body})")
    for eff, vT, T in zip(eff_T_a, vT_a, T_a):
        lines.append(f"{body}if HAS_V_WEIGHTS:")
        lines.append(f"{body}    {eff} = w * {vT} + vw * {T}")
        lines.append(f"{body}else:")
        lines.append(f"{body}    {eff} = w * {vT}")
    for eff, vT, T in zip(eff_T_b, vT_b, T_b):
        lines.append(f"{body}if HAS_V_WEIGHTS:")
        lines.append(f"{body}    {eff} = w * {vT} + vw * {T}")
        lines.append(f"{body}else:")
        lines.append(f"{body}    {eff} = w * {vT}")
    if not swap:
        in1_helper, in1_lhs, in1_inputs = "a", d_T_a, eff_T_b
        in2_helper, in2_lhs, in2_inputs = "b", d_T_b, eff_T_a
    else:
        in1_helper, in1_lhs, in1_inputs = "b", d_T_b, eff_T_a
        in2_helper, in2_lhs, in2_inputs = "a", d_T_a, eff_T_b
    lines.append(f"{body}if NEED_D_IN1:")
    din_body = body + "    "
    lines.append(f"{din_body}{', '.join(in1_lhs)} = accum_grad_l{l1}_l{l2}_to_l{l_out}_{in1_helper}(")
    for nm in grad_out_names:
        lines.append(f"{din_body}    {nm},")
    for nm in in1_inputs:
        lines.append(f"{din_body}    {nm},")
    for nm in in1_lhs[:-1]:
        lines.append(f"{din_body}    {nm},")
    lines.append(f"{din_body}    {in1_lhs[-1]},")
    lines.append(f"{din_body})")
    lines.append(f"{body}if NEED_D_IN2:")
    din_body = body + "    "
    lines.append(f"{din_body}{', '.join(in2_lhs)} = accum_grad_l{l1}_l{l2}_to_l{l_out}_{in2_helper}(")
    for nm in grad_out_names:
        lines.append(f"{din_body}    {nm},")
    for nm in in2_inputs:
        lines.append(f"{din_body}    {nm},")
    for nm in in2_lhs[:-1]:
        lines.append(f"{din_body}    {nm},")
    lines.append(f"{din_body}    {in2_lhs[-1]},")
    lines.append(f"{din_body})")

    lines.append(f"{outer}global_path_idx += 1")
    lines.append(f"{outer}path{l_out} += 1")
    return "\n".join(lines) + "\n"


def _emit_tp_dbwd_kernel(kernel_l_max: int) -> str:
    pointers = [
        "grad_out_ptr",
        "in1_ptr",
        "in2_ptr",
        "weights_ptr",
        "v_in1_ptr",
        "v_in2_ptr",
        "v_weights_ptr",
        "d_grad_out_ptr",
        "d_weights_ptr",
        "d_in1_ptr",
        "d_in2_ptr",
        "idx_i_ptr",
        "idx_j_ptr",
    ]
    sig = _emit_kernel_signature(
        "tp_dbwd_kernel",
        pointers=pointers,
        int_args_before_features=["n_batch"],
        int_args_after_features=[
            "in1_stride",
            "in2_stride",
            "d_grad_out_stride",
            "d_in1_stride",
            "d_in2_stride",
        ],
        kernel_l_max=kernel_l_max,
        include_out_l_sizes=True,
        extra_constexpr_args=[
            "GRAD_OUT_STRIDE",
            "GRAD_OUT_COL_STRIDE",
            "NEED_D_GRAD_OUT",
            "NEED_D_IN1",
            "NEED_D_IN2",
            "NEED_D_WEIGHTS",
            "HAS_V_WEIGHTS",
        ],
    )
    ind = "    "
    body: List[str] = []
    body.append(_emit_kernel_setup_pid_block(ind))
    body.append(f"\n{ind}m1 = b_mask[:, None] & f1_mask[None, :]\n")
    body.append(f"{ind}m2 = b_mask[:, None] & f2_mask[None, :]\n")
    body.append(f"{ind}mo = b_mask[:, None] & f_mask[None, :]\n\n")
    body.append(f"{ind}if NEEDS_SCATTER:\n")
    body.append(f"{ind}    idx_j = tl.load(idx_j_ptr + b, mask=b_mask, other=0)\n")
    body.append(f"{ind}    idx_i = tl.load(idx_i_ptr + b, mask=b_mask, other=0)\n")
    body.append(f"{ind}    in1_row = idx_j[:, None].to(tl.int64) * in1_stride\n")
    body.append(f"{ind}    in2_row = b[:, None].to(tl.int64) * in2_stride\n")
    body.append(f"{ind}    grad_out_row = idx_i[:, None].to(tl.int64) * GRAD_OUT_STRIDE\n")
    body.append(f"{ind}    d_grad_out_row = idx_i[:, None].to(tl.int64) * d_grad_out_stride\n")
    body.append(f"{ind}    d_in1_row = idx_j[:, None].to(tl.int64) * d_in1_stride\n")
    body.append(f"{ind}    d_in2_row = b[:, None].to(tl.int64) * d_in2_stride\n")
    body.append(f"{ind}else:\n")
    body.append(f"{ind}    in1_row = b[:, None].to(tl.int64) * in1_stride\n")
    body.append(f"{ind}    in2_row = b[:, None].to(tl.int64) * in2_stride\n")
    body.append(f"{ind}    grad_out_row = b[:, None].to(tl.int64) * GRAD_OUT_STRIDE\n")
    body.append(f"{ind}    d_grad_out_row = b[:, None].to(tl.int64) * d_grad_out_stride\n")
    body.append(f"{ind}    d_in1_row = b[:, None].to(tl.int64) * d_in1_stride\n")
    body.append(f"{ind}    d_in2_row = b[:, None].to(tl.int64) * d_in2_stride\n")
    body.append("\n")
    body.append(f"{ind}f_idx = f[None, :]\n")
    body.append(f"{ind}f1_idx = f1[None, :]\n")
    body.append(f"{ind}f2_idx = f2[None, :]\n\n")
    body.append(
        _emit_input_loads_block(
            "1",
            "in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "m1",
            "IN1_L_MAX",
            kernel_l_max,
            indent=ind,
        )
    )
    body.append(
        _emit_input_loads_block(
            "1",
            "v_in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "m1",
            "IN1_L_MAX",
            kernel_l_max,
            name_prefix="v",
            indent=ind,
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "m2",
            "IN2_L_MAX",
            kernel_l_max,
            indent=ind,
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "v_in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "m2",
            "IN2_L_MAX",
            kernel_l_max,
            name_prefix="v",
            indent=ind,
        )
    )
    body.append("\n")
    body.append(f"{ind}if NEED_D_IN1:\n")
    body.append(_emit_grad_zeros_block("1", "IN1_L_MAX", kernel_l_max, indent=ind + "    "))
    body.append(f"{ind}if NEED_D_IN2:\n")
    body.append(_emit_grad_zeros_block("2", "IN2_L_MAX", kernel_l_max, indent=ind + "    "))
    body.append("\n")
    body.append(_emit_out_off_constexpr_block(kernel_l_max, indent=ind))
    body.append(f"\n{ind}global_path_idx = 0\n")
    body.append(
        _emit_path_ladder(
            kernel_l_max,
            _emit_dbwd_path_branch,
            ind,
            emit_preamble=lambda l_out: (
                f"{ind}    if NEED_D_WEIGHTS or NEED_D_IN1 or NEED_D_IN2:\n"
                + _emit_bwd_preload_grad_l_out(l_out, indent=ind + "        ")
            ),
        )
    )
    body.append("\n")
    body.append(f"{ind}if NEED_D_IN1:\n")
    body.append(
        _emit_store_grad_in_block(
            "1",
            "d_in1_ptr",
            "d_in1_row",
            "IN1_OFF",
            "f1",
            "b",
            "d_in1_stride",
            "IN1_FEATURES",
            "f_mask",
            "b_mask",
            "m1",
            "NEEDS_SCATTER",
            "IN1_L_MAX",
            kernel_l_max,
            indent=ind + "    ",
        )
    )
    body.append("\n")
    body.append(f"{ind}if NEED_D_IN2:\n")
    body.append(
        _emit_store_grad_in_block(
            "2",
            "d_in2_ptr",
            "d_in2_row",
            "IN2_OFF",
            "f2",
            "b",
            "d_in2_stride",
            "IN2_FEATURES",
            "f_mask",
            "b_mask",
            "m2",
            "False",
            "IN2_L_MAX",
            kernel_l_max,
            indent=ind + "    ",
        )
    )
    return sig + "".join(body)


def _emit_load_weight_csr() -> str:
    return (
        "@triton.jit\n"
        "def load_weight_csr(\n"
        "    w_ptr,\n"
        "    edge,\n"
        "    f_idx,\n"
        "    PATH_IDX: tl.constexpr,\n"
        "    N_TOTAL_PATHS: tl.constexpr,\n"
        "    OUT_FEATURES: tl.constexpr,\n"
        "    SHARED_WEIGHTS: tl.constexpr,\n"
        "    f_mask,\n"
        "    HAS_MASK: tl.constexpr,\n"
        "):\n"
        "    if SHARED_WEIGHTS:\n"
        "        if HAS_MASK:\n"
        "            return tl.load(w_ptr + PATH_IDX * OUT_FEATURES + f_idx, mask=f_mask, other=1.0)\n"
        "        else:\n"
        "            return tl.load(w_ptr + PATH_IDX * OUT_FEATURES + f_idx)\n"
        "    else:\n"
        "        if HAS_MASK:\n"
        "            return tl.load(\n"
        "                w_ptr + edge.to(tl.int64) * (N_TOTAL_PATHS * OUT_FEATURES)\n"
        "                + PATH_IDX * OUT_FEATURES + f_idx,\n"
        '                mask=f_mask, other=1.0, eviction_policy="evict_first",\n'
        "            )\n"
        "        else:\n"
        "            return tl.load(\n"
        "                w_ptr + edge.to(tl.int64) * (N_TOTAL_PATHS * OUT_FEATURES)\n"
        "                + PATH_IDX * OUT_FEATURES + f_idx,\n"
        '                eviction_policy="evict_first",\n'
        "            )\n"
    )


def _emit_store_csr_l(l: int) -> str:
    suffixes = ["v"] if l == 0 else [_cartesian_suffix(a, b, c) for (a, b, c) in stored_basis(l)]
    return _emit_store_kernel(
        f"store_csr_l{l}",
        suffixes,
        "f_mask",
        include_needs_scatter=False,
        reduce_atomic=False,
    )


def _csr_path_id(l1: int, l2: int, l_out: int, swap: bool) -> str:
    return f"{l1}{l2}{l_out}{'s' if swap else 'f'}"


def _csr_path_guard(l1: int, l2: int, swap: bool) -> str:
    if swap:
        return f"IN1_L_MAX >= {l2} and IN2_L_MAX >= {l1}"
    return f"IN1_L_MAX >= {l1} and IN2_L_MAX >= {l2}"


def _csr_acc_names(l1: int, l2: int, l_out: int, swap: bool, prefix: str = "acc") -> List[str]:
    pid = _csr_path_id(l1, l2, l_out, swap)
    return [f"{prefix}{pid}_{c}" for c in _out_result_vars(l_out)]


def _csr_reduce_acc_names(l_out: int, prefix: str = "acc") -> List[str]:
    return [f"{prefix}R{l_out}_{c}" for c in _out_result_vars(l_out)]


def _csr_ordered_paths(kernel_l_max: int) -> List[Tuple[int, int, int, bool]]:
    paths = _tp_path_order(kernel_l_max)
    ordered: List[Tuple[int, int, int, bool]] = []
    for l_out in range(kernel_l_max + 1):
        l_out_paths = [p for p in paths if p[2] == l_out]
        for l1, l2, _lo in l_out_paths:
            ordered.append((l1, l2, l_out, False))
        for l1, l2, _lo in l_out_paths:
            if l1 != l2:
                ordered.append((l1, l2, l_out, True))
    return ordered


def _emit_csr_acc_init_block(
    kernel_l_max: int,
    indent: str,
    prefix: str = "acc",
    ptr: str = "out_ptr",
    shape: str = "[FEATURE_BLOCK]",
) -> str:
    lines: List[str] = []
    lines.append(f"{indent}if REDUCE_PATHS:")
    for l_out in range(kernel_l_max + 1):
        names = _csr_reduce_acc_names(l_out, prefix)
        lines.append(f"{indent}    if OUT_L_MAX >= {l_out} and GROUP_L{l_out} == PATH_GROUP:")
        for nm in names:
            lines.append(f"{indent}        {nm} = tl.zeros({shape}, dtype={ptr}.dtype.element_ty)")
    lines.append(f"{indent}else:")
    for l1, l2, l_out, swap in _csr_ordered_paths(kernel_l_max):
        guard = _csr_path_guard(l1, l2, swap)
        if swap:
            guard = f"not SYMMETRIC_PRODUCT and {guard}"
        lines.append(f"{indent}    if OUT_L_MAX >= {l_out} and GROUP_L{l_out} == PATH_GROUP and {guard}:")
        for nm in _csr_acc_names(l1, l2, l_out, swap, prefix):
            lines.append(f"{indent}        {nm} = tl.zeros({shape}, dtype={ptr}.dtype.element_ty)")
    return "\n".join(lines) + "\n"


def _emit_csr_fwd_one_path(l1: int, l2: int, l_out: int, swap: bool, indent: str) -> str:
    guard = _csr_path_guard(l1, l2, swap)
    if swap:
        guard = f"not SYMMETRIC_PRODUCT and {guard}"
    body = indent + "        "
    lines: List[str] = []
    lines.append(f"{indent}if {guard}:")
    lines.append(f"{indent}    if GROUP_L{l_out} == PATH_GROUP:")
    args_a, args_b = _path_inputs_for_kernel(l1, l2, l_out, swap)
    args = args_a + args_b
    out_vars = _out_result_vars(l_out)
    lhs = ", ".join(out_vars)
    lines.append(f"{body}{lhs} = contract_l{l1}_l{l2}_to_l{l_out}(")
    for a in args:
        lines.append(f"{body}    {a},")
    lines.append(f"{body})")
    lines.append(f"{body}w = load_weight_csr(")
    lines.append(f"{body}    weights_ptr, edge, f_idx, global_path_idx,")
    lines.append(f"{body}    N_TOTAL_PATHS, OUT_FEATURES, SHARED_WEIGHTS, f_mask, HAS_FM,")
    lines.append(f"{body})")
    acc_names = _csr_acc_names(l1, l2, l_out, swap)
    red_names = _csr_reduce_acc_names(l_out)
    lines.append(f"{body}if REDUCE_PATHS:")
    for rn, ov in zip(red_names, out_vars):
        lines.append(f"{body}    {rn} = {rn} + {ov} * w")
    lines.append(f"{body}else:")
    for an, ov in zip(acc_names, out_vars):
        lines.append(f"{body}    {an} = {an} + {ov} * w")
    lines.append(f"{indent}    global_path_idx += 1")
    return "\n".join(lines) + "\n"


def _emit_csr_fwd_loop_paths(kernel_l_max: int, indent: str) -> str:
    return f"{indent}global_path_idx = 0\n" + _emit_path_ladder(
        kernel_l_max,
        _emit_csr_fwd_one_path,
        indent,
        lead_blank=False,
        emit_path_zero=False,
        swap_block=False,
    )


def _emit_csr_store_block(
    kernel_l_max: int,
    indent: str,
    prefix: str = "acc",
    ptr: str = "out_ptr",
    row_var: str = "out_row",
    row_expr: str = "pid_n.to(tl.int64) * out_stride",
    emit_offsets: bool = True,
    path_counter_suffix: str = "",
    mask_var: str = "f_mask",
) -> str:
    lines: List[str] = []
    lines.append(f"{indent}{row_var} = {row_expr}")
    if emit_offsets:
        lines.append(_emit_out_off_constexpr_block(kernel_l_max, indent=indent).rstrip("\n"))
    pc = path_counter_suffix
    for l_out in range(kernel_l_max + 1):
        lines.append(f"{indent}if OUT_L_MAX >= {l_out} and GROUP_L{l_out} == PATH_GROUP:")
        red_names = _csr_reduce_acc_names(l_out, prefix)
        lines.append(f"{indent}    if REDUCE_PATHS:")
        # Guard on n_paths: an l_out with no allowed paths has a zero-size
        # output block, so its (reduced) store must be pruned or it writes
        # past the block into the next node's row.
        lines.append(f"{indent}        if N_PATHS_L{l_out} > 0:")
        lines.append(f"{indent}            store_csr_l{l_out}(")
        lines.append(f"{indent}                {ptr}, {row_var}, OUT_L{l_out}_OFF, f_idx,")
        lines.append(f"{indent}                OUT_FEATURES, N_PATHS_L{l_out}, 0,")
        for rn in red_names:
            lines.append(f"{indent}                {rn},")
        lines.append(f"{indent}                {mask_var}, REDUCE_PATHS,")
        lines.append(f"{indent}            )")
        lines.append(f"{indent}    else:")
        lines.append(f"{indent}        path{l_out}{pc} = 0")
        paths_lo = [p for p in _csr_ordered_paths(kernel_l_max) if p[2] == l_out]
        for l1, l2, _lo, swap in paths_lo:
            guard = _csr_path_guard(l1, l2, swap)
            if swap:
                guard = f"not SYMMETRIC_PRODUCT and {guard}"
            body = indent + "            "
            lines.append(f"{indent}        if {guard}:")
            lines.append(f"{body}store_csr_l{l_out}(")
            lines.append(f"{body}    {ptr}, {row_var}, OUT_L{l_out}_OFF, f_idx,")
            lines.append(f"{body}    OUT_FEATURES, N_PATHS_L{l_out}, path{l_out}{pc},")
            for an in _csr_acc_names(l1, l2, l_out, swap, prefix):
                lines.append(f"{body}    {an},")
            lines.append(f"{body}    {mask_var}, REDUCE_PATHS,")
            lines.append(f"{body})")
            lines.append(f"{body}path{l_out}{pc} += 1")
    return "\n".join(lines) + "\n"


def _emit_csr_setup_block(
    pid_lines: Sequence[str] = ("pid_n = tl.program_id(0)", "pid_f = tl.program_id(1)"),
    indent: str = "    ",
    two_d: bool = False,
) -> str:
    parts: List[str] = [f"{indent}{ln}\n" for ln in pid_lines]
    if not two_d:
        parts.append(
            f"\n"
            f"{indent}HAS_FM: tl.constexpr = OUT_FEATURES % FEATURE_BLOCK != 0\n"
            f"{indent}HAS_M1: tl.constexpr = HAS_FM and IN1_FEATURES != 1\n"
            f"{indent}HAS_M2: tl.constexpr = HAS_FM and IN2_FEATURES != 1\n"
            f"{indent}f = pid_f * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)\n"
            f"{indent}if HAS_FM:\n"
            f"{indent}    f_mask = f < OUT_FEATURES\n"
            f"{indent}else:\n"
            f"{indent}    f_mask = None\n"
            f"\n"
            f"{indent}if IN1_FEATURES == 1:\n"
            f"{indent}    f1 = tl.zeros([FEATURE_BLOCK], dtype=tl.int32)\n"
            f"{indent}else:\n"
            f"{indent}    f1 = f\n"
            f"{indent}if HAS_M1:\n"
            f"{indent}    f1_mask = f_mask\n"
            f"{indent}else:\n"
            f"{indent}    f1_mask = None\n"
            f"\n"
            f"{indent}if IN2_FEATURES == 1:\n"
            f"{indent}    f2 = tl.zeros([FEATURE_BLOCK], dtype=tl.int32)\n"
            f"{indent}else:\n"
            f"{indent}    f2 = f\n"
            f"{indent}if HAS_M2:\n"
            f"{indent}    f2_mask = f_mask\n"
            f"{indent}else:\n"
            f"{indent}    f2_mask = None\n"
            f"\n"
            f"{indent}f_idx = f\n"
            f"{indent}f1_idx = f1\n"
            f"{indent}f2_idx = f2\n"
            f"\n"
            f"{indent}row_start = tl.load(rowptr_ptr + pid_n)\n"
            f"{indent}row_end = tl.load(rowptr_ptr + pid_n + 1)\n"
        )
        return "".join(parts)
    parts.append(
        f"\n"
        f"{indent}f = pid_f * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)\n"
        f"{indent}f_mask = f < OUT_FEATURES\n"
        f"\n"
        f"{indent}if IN1_FEATURES == 1:\n"
        f"{indent}    f1 = tl.zeros([FEATURE_BLOCK], dtype=tl.int32)\n"
        f"{indent}    f1_mask = tl.full([FEATURE_BLOCK], True, tl.int1)\n"
        f"{indent}else:\n"
        f"{indent}    f1 = f\n"
        f"{indent}    f1_mask = f_mask\n"
        f"{indent}if IN2_FEATURES == 1:\n"
        f"{indent}    f2 = tl.zeros([FEATURE_BLOCK], dtype=tl.int32)\n"
        f"{indent}    f2_mask = tl.full([FEATURE_BLOCK], True, tl.int1)\n"
        f"{indent}else:\n"
        f"{indent}    f2 = f\n"
        f"{indent}    f2_mask = f_mask\n"
        f"\n"
        f"{indent}b_mask = tl.full([1], True, tl.int1)\n"
        f"{indent}f_idx = f[None, :]\n"
        f"{indent}f1_idx = f1[None, :]\n"
        f"{indent}f2_idx = f2[None, :]\n"
        f"{indent}mo = b_mask[:, None] & f_mask[None, :]\n"
        f"{indent}m1 = b_mask[:, None] & f1_mask[None, :]\n"
        f"{indent}m2 = b_mask[:, None] & f2_mask[None, :]\n"
        f"\n"
    )
    return "".join(parts)


def _emit_tp_fwd_csr_kernel(kernel_l_max: int) -> str:
    sig = _emit_kernel_signature(
        "tp_fwd_csr_kernel",
        pointers=["in1_ptr", "in2_ptr", "out_ptr", "weights_ptr", "idx_j_ptr", "rowptr_ptr"],
        int_args_before_features=[
            "in1_stride",
            "in2_stride",
            "out_stride",
        ],
        int_args_after_features=[],
        kernel_l_max=kernel_l_max,
        include_out_l_sizes=True,
        extra_constexpr_args=[
            "PATH_GROUP",
            *[f"GROUP_L{l}" for l in range(kernel_l_max + 1)],
        ],
        include_needs_scatter=False,
        include_block_size=False,
    )

    ind = "    "
    body: List[str] = []
    body.append(_emit_csr_setup_block(indent=ind))
    body.append("\n")
    body.append(_emit_csr_acc_init_block(kernel_l_max, ind))
    body.append("\n")
    body.append(f"{ind}for edge in range(row_start, row_end):\n")
    loop = ind + "    "
    body.append(f"{loop}j = tl.load(idx_j_ptr + edge)\n")
    body.append(f"{loop}in1_row = j.to(tl.int64) * in1_stride\n")
    body.append(f"{loop}in2_row = edge.to(tl.int64) * in2_stride\n")
    body.append(
        _emit_input_loads_block(
            "1",
            "in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "f1_mask",
            "IN1_L_MAX",
            kernel_l_max,
            indent=loop,
            evict="evict_last",
            has_mask="HAS_M1",
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "f2_mask",
            "IN2_L_MAX",
            kernel_l_max,
            indent=loop,
            evict="evict_first",
            has_mask="HAS_M2",
        )
    )
    body.append(_emit_csr_fwd_loop_paths(kernel_l_max, loop))
    body.append("\n")
    body.append(_emit_csr_store_block(kernel_l_max, ind))
    return sig + "".join(body)


def _emit_tp_dbwd_csr_kernel(kernel_l_max: int) -> str:
    sig = _emit_kernel_signature(
        "tp_dbwd_csr_kernel",
        pointers=[
            "grad_out_ptr",
            "in1_ptr",
            "in2_ptr",
            "weights_ptr",
            "v_in1_ptr",
            "v_in2_ptr",
            "v_weights_ptr",
            "d_grad_out_ptr",
            "d_weights_ptr",
            "d_in1_ptr",
            "d_in2_ptr",
            "idx_j_ptr",
            "rowptr_ptr",
        ],
        int_args_before_features=[
            "in1_stride",
            "in2_stride",
            "d_grad_out_stride",
            "d_in1_stride",
            "d_in2_stride",
        ],
        int_args_after_features=[],
        kernel_l_max=kernel_l_max,
        include_out_l_sizes=True,
        extra_constexpr_args=[
            "GRAD_OUT_STRIDE",
            "GRAD_OUT_COL_STRIDE",
            "NEED_D_GRAD_OUT",
            "NEED_D_IN1",
            "NEED_D_IN2",
            "NEED_D_WEIGHTS",
            "HAS_V_WEIGHTS",
            "PATH_GROUP",
            "N_PATH_GROUPS",
            *[f"GROUP_L{l}" for l in range(kernel_l_max + 1)],
        ],
        include_needs_scatter=False,
        include_block_size=False,
    )

    ind = "    "
    body: List[str] = []
    body.append(_emit_csr_setup_block(indent=ind, two_d=True))
    body.append(_emit_out_off_constexpr_block(kernel_l_max, indent=ind))
    body.append(f"{ind}if NEED_D_IN1 or NEED_D_IN2 or NEED_D_WEIGHTS:\n")
    body.append(f"{ind}    grad_out_row = (pid_n.to(tl.int64) * GRAD_OUT_STRIDE) + tl.zeros([1, 1], tl.int64)\n")
    body.append(_emit_bwd_csr_grad_out_preload(kernel_l_max, ind + "    "))
    body.append(f"{ind}if NEED_D_GRAD_OUT:\n")
    body.append(
        _emit_csr_acc_init_block(
            kernel_l_max,
            ind + "    ",
            prefix="dgo",
            ptr="d_grad_out_ptr",
            shape="[1, FEATURE_BLOCK]",
        )
    )
    body.append("\n")
    body.append(f"{ind}row_start = tl.load(rowptr_ptr + pid_n)\n")
    body.append(f"{ind}row_end = tl.load(rowptr_ptr + pid_n + 1)\n")
    body.append(f"{ind}for edge in range(row_start, row_end):\n")
    loop = ind + "    "
    body.append(f"{loop}j = tl.load(idx_j_ptr + edge)\n")
    body.append(f"{loop}b = edge + tl.zeros([1], tl.int32)\n")
    body.append(f"{loop}in1_row = j.to(tl.int64) * in1_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(f"{loop}in2_row = edge.to(tl.int64) * in2_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(f"{loop}d_in1_row = j.to(tl.int64) * d_in1_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(f"{loop}d_in2_row = edge.to(tl.int64) * d_in2_stride + tl.zeros([1, 1], tl.int64)\n")
    body.append(
        _emit_input_loads_block(
            "1",
            "in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "m1",
            "IN1_L_MAX",
            kernel_l_max,
            indent=loop,
        )
    )
    body.append(
        _emit_input_loads_block(
            "1",
            "v_in1_ptr",
            "in1_row",
            "IN1_OFF",
            "IN1_FEATURES",
            "f1_idx",
            "m1",
            "IN1_L_MAX",
            kernel_l_max,
            name_prefix="v",
            indent=loop,
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "m2",
            "IN2_L_MAX",
            kernel_l_max,
            indent=loop,
        )
    )
    body.append(
        _emit_input_loads_block(
            "2",
            "v_in2_ptr",
            "in2_row",
            "IN2_OFF",
            "IN2_FEATURES",
            "f2_idx",
            "m2",
            "IN2_L_MAX",
            kernel_l_max,
            name_prefix="v",
            indent=loop,
        )
    )
    body.append(f"{loop}if NEED_D_IN1:\n")
    body.append(_emit_grad_zeros_block("1", "IN1_L_MAX", kernel_l_max, indent=loop + "    "))
    body.append(f"{loop}if NEED_D_IN2:\n")
    body.append(_emit_grad_zeros_block("2", "IN2_L_MAX", kernel_l_max, indent=loop + "    "))
    body.append(f"{loop}global_path_idx = 0\n")
    body.append(
        _emit_path_ladder(
            kernel_l_max,
            lambda l1, l2, l_out, swap, indent: _emit_dbwd_path_branch(
                l1,
                l2,
                l_out,
                swap=swap,
                indent=indent,
                group=True,
                preloaded=True,
                accumulate_d_grad_out=True,
            ),
            loop,
            lead_blank=False,
        )
    )
    body.append(f"{loop}if NEED_D_IN1:\n")
    body.append(
        _emit_store_grad_in_block(
            "1",
            "d_in1_ptr",
            "d_in1_row",
            "IN1_OFF",
            "f1",
            "b",
            "d_in1_stride",
            "IN1_FEATURES",
            "f_mask",
            "b_mask",
            "m1",
            "True",
            "IN1_L_MAX",
            kernel_l_max,
            indent=loop + "    ",
        )
    )
    body.append(f"{loop}if NEED_D_IN2:\n")
    body.append(
        _emit_store_grad_in_block(
            "2",
            "d_in2_ptr",
            "d_in2_row",
            "IN2_OFF",
            "f2",
            "b",
            "d_in2_stride",
            "IN2_FEATURES",
            "f_mask",
            "b_mask",
            "m2",
            "N_PATH_GROUPS > 1",
            "IN2_L_MAX",
            kernel_l_max,
            indent=loop + "    ",
        )
    )
    body.append("\n")
    body.append(f"{ind}if NEED_D_GRAD_OUT:\n")
    body.append(
        _emit_csr_store_block(
            kernel_l_max,
            ind + "    ",
            prefix="dgo",
            ptr="d_grad_out_ptr",
            row_var="d_grad_out_row",
            row_expr="pid_n.to(tl.int64) * d_grad_out_stride + tl.zeros([1, 1], tl.int64)",
            emit_offsets=False,
            path_counter_suffix="_s",
            mask_var="mo",
        )
    )
    return sig + "".join(body)


def build_triton_tp_module_source(kernel_l_max: int) -> str:
    """Generate a Triton tensor-product module as Python source.

    Args:
        kernel_l_max (int): Maximum tensor rank included in the generated module.

    Returns:
        str: Python module source. The cache layer adds the fingerprint header
            when writing the module.
    """
    if kernel_l_max < 0:
        raise ValueError(f"kernel_l_max must be non-negative, got {kernel_l_max}.")
    paths = _tp_path_order(kernel_l_max)

    pieces: List[str] = []
    pieces.append(
        '"""Auto-generated by flashcart.o3._codegen_tensor_product.build_triton_tp_module_source.\n'
        "\n"
        "Do not edit: regenerated whenever the codegen fingerprint changes.\n"
        "Naming: tl.constexpr params/locals are UPPER_SNAKE; runtime args and\n"
        "locals lower_snake. acc{l1}{l2}{lo}{f|s}_* = per-path CSR accumulators\n"
        "(f=canonical, s=swapped operand order); go*/dgo* = per-receiver\n"
        "grad_out preload / d_grad_out accumulators; uc/uab/uab2 = dbwd\n"
        "contraction intermediates; eff_* = w*vT + vw*T effective inputs;\n"
        "accum_grad_*_{a,b} = single-operand backward helpers. See the\n"
        'codegen module docstring for the full legend."""\n'
    )
    pieces.append("\n")
    pieces.append("import triton\n")
    pieces.append("import triton.language as tl  # noqa: F401  (referenced by @triton.jit)\n")
    pieces.append("\n")
    pieces.append(f"KERNEL_L_MAX = {kernel_l_max}\n")
    pieces.append("\n")
    pieces.append(_TP_AUTOTUNE_HEADER)
    pieces.append("\n")

    pieces.append(_emit_low_level_helpers(kernel_l_max))

    for path in paths:
        pieces.append("\n")
        pieces.append(_emit_contract_function(*path))
        pieces.append("\n")

    for path in paths:
        pieces.append("\n")
        pieces.append(_emit_accum_grad_function(*path))
        pieces.append("\n")

    for path in paths:
        for side in ("a", "b"):
            pieces.append("\n")
            pieces.append(_emit_accum_grad_function_one_side(*path, side=side))
            pieces.append("\n")

    pieces.append("@triton.autotune(\n")
    pieces.append("    configs=_TP_FWD_AUTOTUNE_CONFIGS,\n")
    pieces.append("    key=_TP_FWD_AUTOTUNE_KEYS,\n")
    pieces.append('    reset_to_zero=["out_ptr"],\n')
    pieces.append(")\n")
    pieces.append(_emit_tp_fwd_kernel(kernel_l_max))
    pieces.append("\n\n")
    pieces.append("\n# --- CSR (segmented) forward kernel ---\n\n")
    pieces.append(_emit_load_weight_csr())
    pieces.append("\n")
    for l in range(kernel_l_max + 1):
        pieces.append(_emit_store_csr_l(l))
        pieces.append("\n")
    pieces.append("@triton.autotune(\n")
    pieces.append("    configs=_TP_CSR_AUTOTUNE_CONFIGS,\n")
    pieces.append("    key=_TP_FWD_CSR_AUTOTUNE_KEYS,\n")
    pieces.append(")\n")
    pieces.append(_emit_tp_fwd_csr_kernel(kernel_l_max))
    pieces.append("\n\n")
    pieces.append("\n# --- CSR (segment-walk) backward kernel: grad_out gather amortized ---\n\n")
    pieces.append("@triton.autotune(\n")
    pieces.append("    configs=_TP_BWD_CSR_AUTOTUNE_CONFIGS,\n")
    pieces.append(
        '    key=_TP_FWD_CSR_AUTOTUNE_KEYS + ["NEED_GRAD_IN1", "NEED_GRAD_IN2", '
        '"NEED_GRAD_WEIGHTS", "PATH_GROUP", "N_CHUNK_SLOTS"],\n'
    )
    pieces.append('    reset_to_zero=["grad_in1_ptr", "grad_in2_ptr", "grad_weights_ptr"],\n')
    pieces.append(")\n")
    pieces.append(_emit_tp_bwd_csr_kernel(kernel_l_max))
    pieces.append("\n\n")
    pieces.append("\n# --- fused backward kernel ---\n\n")
    pieces.append("@triton.autotune(\n")
    pieces.append("    configs=_TP_BWD_AUTOTUNE_CONFIGS,\n")
    pieces.append("    key=_TP_BWD_AUTOTUNE_KEYS,\n")
    pieces.append('    reset_to_zero=["grad_in1_ptr", "grad_in2_ptr", "grad_weights_ptr"],\n')
    pieces.append(")\n")
    pieces.append(_emit_tp_bwd_kernel(kernel_l_max))
    pieces.append("\n\n")
    pieces.append("\n# --- fused double-backward kernel ---\n\n")
    pieces.append("@triton.autotune(\n")
    pieces.append("    configs=_TP_DBWD_AUTOTUNE_CONFIGS,\n")
    pieces.append("    key=_TP_DBWD_AUTOTUNE_KEYS,\n")
    pieces.append('    reset_to_zero=["d_grad_out_ptr", "d_weights_ptr", "d_in1_ptr", "d_in2_ptr"],\n')
    pieces.append(")\n")
    pieces.append(_emit_tp_dbwd_kernel(kernel_l_max))
    pieces.append("\n\n")
    pieces.append("\n# --- CSR (segment-walk) double-backward kernel ---\n\n")
    pieces.append("@triton.autotune(\n")
    pieces.append("    configs=_TP_CSR_AUTOTUNE_CONFIGS,\n")
    pieces.append(
        '    key=_TP_FWD_CSR_AUTOTUNE_KEYS + ["NEED_D_GRAD_OUT", "NEED_D_IN1", '
        '"NEED_D_IN2", "NEED_D_WEIGHTS", "HAS_V_WEIGHTS"],\n'
    )
    pieces.append('    reset_to_zero=["d_in1_ptr", "d_in2_ptr", "d_weights_ptr"],\n')
    pieces.append(")\n")
    pieces.append(_emit_tp_dbwd_csr_kernel(kernel_l_max))
    pieces.append("\n\n")

    return "".join(pieces)
