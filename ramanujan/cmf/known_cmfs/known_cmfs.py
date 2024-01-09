import sympy as sp
from sympy.abc import x, y

from ramanujan import Matrix
from ramanujan.cmf import CMF, ffbar

c0, c1, c2, c3, c4 = sp.symbols("c:5")


def e():
    return CMF(
        Matrix([[1, -y - 1], [-1, x + y + 2]]), Matrix([[0, -y - 1], [-1, x + y + 1]])
    )


def pi():
    return CMF(
        Matrix([[x, -x], [-y, 2 * x + y + 1]]),
        Matrix([[1 + y, -x], [-1 - y, x + 2 * y + 1]]),
    )


def zeta3():
    return CMF(
        Matrix(
            [
                [0, -(x**3)],
                [(x + 1) ** 3, x**3 + (x + 1) ** 3 + 2 * y * (y - 1) * (2 * x + 1)],
            ]
        ),
        Matrix(
            [
                [-(x**3) + 2 * x**2 * y - 2 * x * y**2 + y**3, -(x**3)],
                [x**3, x**3 + 2 * x**2 * y + 2 * x * y**2 + y**3],
            ]
        ),
    )


def var_root_cmf():
    """
    This is not a standard f,bar(f) matrix field.
    Note that b(x,y) depends on y, while a(x) does not, and x=c/2-y is the root which depends on y
    """
    Y = 2 * y - c2
    b = -x * (x + c0) * (x + c1) * (2 * x + Y)
    a = (2 * x + c1 + 1) * (2 * x + c0 + 1) - x * (x + 1)
    F = x**2 + x * (Y + 1) + (Y + 1 - c1) * (Y + 1 - c0)
    G = -(Y + 2 * x) * (x + c1 + c0 - (Y + 1))
    return CMF(Mx=Matrix([[0, b], [1, a]]), My=Matrix([[G, b], [1, F]]))


def cmf1():
    return ffbar.construct(c0 + c1 * (x + y), c2 + c3 * (x - y))


def cmf2():
    return ffbar.construct(
        (2 * c1 + c2) * (c1 + c2)
        - c3 * c0
        - c3 * ((2 * c1 + c2) * (x + y) + (c1 + c2) * (2 * x + y))
        + c3**2 * (2 * x**2 + 2 * x * y + y**2),
        c3 * (c0 + c2 * x + c1 * y) - c3**2 * (2 * x**2 - 2 * x * y + y**2),
    )


def cmf3_1():
    return ffbar.construct(
        -((c0 + c1 * (x + y)) * (c0 * (x + 2 * y) + c1 * (x**2 + x * y + y**2))),
        (c0 + c1 * (-x + y)) * (c0 * (x - 2 * y) - c1 * (x**2 - x * y + y**2)),
    )


def cmf3_2():
    return ffbar.construct(
        -(x + y) * (c0**2 + 2 * c1**2 * (x**2 + x * y + y**2)),
        (x - y) * (c0**2 + 2 * c1**2 * (x**2 - x * y + y**2)),
    )


def cmf3_3():
    return ffbar.construct(
        (x + y)
        * (c0**2 - c0 * c1 * (x - y) - 2 * c1**2 * (x**2 + x * y + y**2)),
        (c0 + c1 * (x - y)) * (3 * c0 * (x - y) + 2 * c1 * (x**2 - x * y + y**2)),
    )
