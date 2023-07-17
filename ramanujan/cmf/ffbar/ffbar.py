import sympy as sp
from sympy.abc import x, y

from ramanujan import Matrix
from ramanujan.cmf import CMF


def linear_condition(f, fbar) -> bool:
    r"""
    Checks if `f` and `fbar` satisfy the linear condition

    Functions $f(x, y), \bar{f}(x, y)$ satisfy the linear condition iff:
    $f(x+1, y-1) - \bar{f}(x, y-1) + \bar{f}(x+1, y) - f(x, y) = 0$
    """
    return (
        sp.simplify(
            f.subs([[x, x + 1], [y, y - 1]])
            - fbar.subs(y, y - 1)
            + fbar.subs(x, x + 1)
            - f
        )
        == 0
    )


def quadratic_condition(f, fbar) -> bool:
    r"""
    Checks if `f` and `fbar` satisfy the quadratic condition

    Functions $f(x, y), \bar{f}(x, y)$ satisfy the quadratic condition iff:
    $f\bar{f}(x, y) - f\bar{f}(x, 0) -f\bar{f}(0, y) + f\bar{f}(0, 0) = 0$,
    where $f\bar{f}(x, y) = f(x, y) \cdot \bar{f}(x, y)$
    """
    ffbar = f * fbar
    return (
        sp.simplify(
            ffbar - ffbar.subs(x, 0) - ffbar.subs(y, 0) + ffbar.subs([[x, 0], [y, 0]])
        )
        == 0
    )


def a(f, fbar) -> sp.Expr:
    r"""
    Returns the $a(x, y)$ function as constructed in the ffbar construction:
    $a(x, y) = f(x, y) - \bar{f}(x+1, y) = f(x+1, y-1) - \bar{f}(x, y-1)$
    """
    return f - fbar.subs(x, x + 1)


def b(f, fbar) -> sp.Expr:
    r"""
    Returns the $b(x)$ function as constructed in the ffbar construction:
    $b(x) = f\bar{f}(x, 0) - f\bar{f}(0, 0) = f\bar{f}(x, y) - f\bar{f}(0, y)$,
    where $f\bar{f}(x, y) = f(x, y) \cdot \bar{f}(x, y)$
    """
    ffbar_x_0 = (f * fbar).subs(y, 0)
    return sp.simplify(ffbar_x_0 - ffbar_x_0.subs(x, 0))


def Mx(f, fbar) -> Matrix:
    r"""
    Returns the $Mx$ matrix as constructed in the ffbar construction:
    $Mx = \begin{pmatrix} 0, b(x) \cr 1, a(x, y) \end{pmatrix}$
    """
    return Matrix([[0, b(f, fbar)], [1, a(f, fbar)]])


def My(f, fbar) -> Matrix:
    r"""
    Returns the $Mx$ matrix as constructed in the ffbar construction:
    $My = \begin{pmatrix} \bar{f}(x, y), b(x) \cr 1, f(x, y) \end{pmatrix}$
    """
    return Matrix([[fbar, b(f, fbar)], [1, f]])


def construct(f, fbar) -> CMF:
    r"""
    Constructs CMF using ffbar construction.

    Asserts that the `f` and `fbar` functions satisfy the linear and quadratic conditions.
    """
    assert linear_condition(f, fbar), (
        "given f and fbar do not satisfy the linear condition! f="
        + str(f)
        + ", fbar="
        + str(fbar)
    )
    assert quadratic_condition(f, fbar), (
        "given f and fbar do not satisfy the quadratic condition! f="
        + str(f)
        + ", fbar="
        + str(fbar)
    )
    return CMF(Mx(f, fbar), My(f, fbar))
