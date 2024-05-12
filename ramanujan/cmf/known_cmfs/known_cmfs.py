import sympy as sp
from sympy.abc import a, b, c, x, y, z

x1 = sp.Symbol("x1")
x2 = sp.Symbol("x2")
x3 = sp.Symbol("x3")

y1 = sp.Symbol("y1")
y2 = sp.Symbol("y2")

from ramanujan import SquareMatrix
from ramanujan.cmf import CMF
from ramanujan.cmf.ffbar import FFbar

c0, c1, c2, c3 = sp.symbols("c:4")


def e():
    return CMF(
        matrices={
            x: SquareMatrix([[1, -y - 1], [-1, x + y + 2]]),
            y: SquareMatrix([[0, -y - 1], [-1, x + y + 1]]),
        }
    )


def pi():
    return CMF(
        matrices={
            x: SquareMatrix([[x, -x], [-y, 2 * x + y + 1]]),
            y: SquareMatrix([[1 + y, -x], [-1 - y, x + 2 * y + 1]]),
        }
    )


def zeta3():
    return CMF(
        matrices={
            x: SquareMatrix(
                [
                    [0, -(x**3)],
                    [(x + 1) ** 3, x**3 + (x + 1) ** 3 + 2 * y * (y - 1) * (2 * x + 1)],
                ]
            ),
            y: SquareMatrix(
                [
                    [-(x**3) + 2 * x**2 * y - 2 * x * y**2 + y**3, -(x**3)],
                    [x**3, x**3 + 2 * x**2 * y + 2 * x * y**2 + y**3],
                ]
            ),
        }
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
    return CMF(
        matrices={x: SquareMatrix([[0, b], [1, a]]), y: SquareMatrix([[G, b], [1, F]])}
    )


def cmf1():
    return FFbar(f=c0 + c1 * (x + y), fbar=c2 + c3 * (x - y))


def cmf2():
    return FFbar(
        f=(
            (2 * c1 + c2) * (c1 + c2)
            - c3 * c0
            - c3 * ((2 * c1 + c2) * (x + y) + (c1 + c2) * (2 * x + y))
            + c3**2 * (2 * x**2 + 2 * x * y + y**2)
        ),
        fbar=c3 * (c0 + c2 * x + c1 * y) - c3**2 * (2 * x**2 - 2 * x * y + y**2),
    )


def cmf3_1():
    return FFbar(
        f=-((c0 + c1 * (x + y)) * (c0 * (x + 2 * y) + c1 * (x**2 + x * y + y**2))),
        fbar=(c0 + c1 * (-x + y)) * (c0 * (x - 2 * y) - c1 * (x**2 - x * y + y**2)),
    )


def cmf3_2():
    return FFbar(
        f=-(x + y) * (c0**2 + 2 * c1**2 * (x**2 + x * y + y**2)),
        fbar=(x - y) * (c0**2 + 2 * c1**2 * (x**2 - x * y + y**2)),
    )


def cmf3_3():
    return FFbar(
        f=(x + y) * (c0**2 - c0 * c1 * (x - y) - 2 * c1**2 * (x**2 + x * y + y**2)),
        fbar=(c0 + c1 * (x - y)) * (3 * c0 * (x - y) + 2 * c1 * (x**2 - x * y + y**2)),
    )


def hypergeometric_dervied_2F1():
    return CMF(
        matrices={
            a: SquareMatrix(
                [[1 + 2 * a, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 4 * a + 2 * b + 4 * c]]
            ),
            b: SquareMatrix(
                [[1 + 2 * b, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 2 * a + 4 * b + 4 * c]]
            ),
            c: SquareMatrix(
                [
                    [-1 - 2 * c, (1 + 2 * a) * (1 + 2 * b)],
                    [1, 3 + 2 * a + 2 * b + 2 * c],
                ]
            ),
        }
    )


def hypergeometric_dervied_3F2():
    return CMF(
        matrices={
            a: SquareMatrix(
                [[1 + 2 * a, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 4 * a + 2 * b + 4 * c]]
            ),
            b: SquareMatrix(
                [[1 + 2 * b, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 2 * a + 4 * b + 4 * c]]
            ),
            c: SquareMatrix(
                [
                    [-1 - 2 * c, (1 + 2 * a) * (1 + 2 * b)],
                    [1, 3 + 2 * a + 2 * b + 2 * c],
                ]
            ),
        }
    )


def hypergeometric_dervied_3F2():
    Sx = x1 + x2 + x3
    Sy = y1 + y2
    Tx = x1 * x2 + x1 * x3 + x2 * x3
    Ty = y1 * y2
    Px = x1 * x2 * x3
    M = SquareMatrix(
        [
            [0, z, 0],
            [0, 1, z],
            [
                Px / (z * (1 - z)),
                ((1 + Sx + Tx) * z - Ty) / (z * (1 - z)),
                1 + (((1 - Sy) + (1 + Sx) * z) / (1 - z)),
            ],
        ],
    )
    I = sp.eye(3)
    return CMF(
        matrices={
            x1: M / x1 + I,
            x2: M / x2 + I,
            x3: M / x3 + I,
            y1: M / (y1 - 1) + I,
            y2: M / (y2 - 1) + I,
        }
    )
