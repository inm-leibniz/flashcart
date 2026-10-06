"""Share component ordering and expression rendering between tensor generators.

``stored_basis`` defines the independent Cartesian components used by the
expansion, tensor-product, and layout-conversion code. Components with two
or more z indices are reconstructed from tracelessness.

``double_factorial`` supplies the combinatorial factors used in normalization.
``_lambdify`` produces numerical reference callables. ``_pycode`` renders
expanded symbolic expressions as Python source and replaces small integer
powers of simple variable names with repeated multiplication.
"""

import re as _re
from typing import List, Sequence, Tuple

import sympy as sp


def double_factorial(n: int) -> int:
    """Compute the double factorial with the generator's negative-input convention.

    Args:
        n (int): Integer argument. Negative values return one.

    Returns:
        int: ``n!!`` for nonnegative ``n``, or one for negative ``n``.
    """
    if n < 0:
        return 1
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


def stored_basis(l: int) -> List[Tuple[int, int, int]]:
    """Return Cartesian index counts for the ``2*l + 1`` stored components.

    Each tuple ``(a, b, c)`` satisfies ``a + b + c == l`` and corresponds to component
    ``T^l_{x...x y...y z...z}`` with ``a`` x's, ``b`` y's, ``c`` z's. Ranks zero through
    three use the explicit order defined here. Higher ranks list ``c == 0`` by
    descending ``a``, followed by ``c == 1`` in the same order.

    Args:
        l (int): Nonnegative tensor rank.

    Returns:
        list[tuple[int, int, int]]: Cartesian index counts in stored-component order.
    """
    if l == 0:
        return [(0, 0, 0)]
    if l == 1:
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if l == 2:
        return [(2, 0, 0), (0, 2, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    if l == 3:
        return [
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (0, 2, 1),
            (0, 3, 0),
        ]
    components: List[Tuple[int, int, int]] = []
    for a in range(l, -1, -1):
        components.append((a, l - a, 0))
    for a in range(l - 1, -1, -1):
        components.append((a, l - 1 - a, 1))
    assert len(components) == 2 * l + 1
    return components


def _lambdify(symbols: Sequence[sp.Symbol], expr):
    """Convert symbolic expressions to a numerical reference callable.

    Args:
        symbols (Sequence[sympy.Symbol]): Callable arguments in their required order.
        expr: Symbolic expression or collection of expressions to evaluate.

    Returns:
        Callable: Function produced by SymPy using the ``math`` namespace. The
            generator's polynomial expressions can also operate on array inputs.
    """
    return sp.lambdify(symbols, expr, modules="math")


_POW_RE = _re.compile(r"([a-zA-Z_]\w*)\*\*(\d+)")


def _expand_int_powers(source: str, max_n: int = 12) -> str:
    """Expand small integer powers of simple variable names into multiplication.

    Args:
        source (str): Python expression source.
        max_n (int, optional): Largest exponent to expand. Default: 12.

    Returns:
        str: Source with powers from two to ``max_n`` expanded. Other powers
            and expressions are left unchanged.
    """
    def repl(m: "_re.Match[str]") -> str:
        base, n_str = m.group(1), m.group(2)
        n = int(n_str)
        if 2 <= n <= max_n:
            return "*".join([base] * n)
        return m.group(0)

    return _POW_RE.sub(repl, source)


def _pycode(expr: sp.Expr) -> str:
    """Render an expanded symbolic expression as Python arithmetic.

    Args:
        expr (sympy.Expr): Expression to expand and print.

    Returns:
        str: Python source with small integer powers written as multiplication.
    """
    return _expand_int_powers(sp.printing.pycode(sp.expand(expr)))
