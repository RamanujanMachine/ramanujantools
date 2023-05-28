import sympy as sp
from sympy.abc import x, y

from ramanujan import Matrix, CMF


def linear_condition(f, fbar) -> bool:
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
    ffbar = f * fbar
    return (
        sp.simplify(
            ffbar - ffbar.subs(x, 0) - ffbar.subs(y, 0) + ffbar.subs([[x, 0], [y, 0]])
        )
        == 0
    )


def a(f, fbar) -> sp.Expr:
    return f - fbar.subs(x, x + 1)


def b(f, fbar) -> sp.Expr:
    ffbar_x_0 = (f * fbar).subs(y, 0)
    return sp.simplify(ffbar_x_0 - ffbar_x_0.subs(x, 0))


def Mx(f, fbar) -> Matrix:
    return Matrix([[0, b(f, fbar)], [1, a(f, fbar)]])


def My(f, fbar) -> Matrix:
    return Matrix([[fbar, b(f, fbar)], [1, f]])


def construct(f, fbar) -> CMF:
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
