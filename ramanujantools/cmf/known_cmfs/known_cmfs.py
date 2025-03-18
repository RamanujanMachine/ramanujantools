import sympy as sp
from sympy.abc import a, b, c, x, y, z

from ramanujantools import Matrix
from ramanujantools.cmf import CMF
from ramanujantools.cmf.ffbar import FFbar

x0, x1, x2 = sp.symbols("x:3")
y0, y1 = sp.symbols("y:2")
c0, c1, c2, c3 = sp.symbols("c:4")
theta = sp.symbols("theta")


def e() -> CMF:
    return CMF(
        matrices={
            x: Matrix([[1, -y - 1], [-1, x + y + 2]]),
            y: Matrix([[0, -y - 1], [-1, x + y + 1]]),
        },
        validate=False,
    )


def pi() -> CMF:
    return CMF(
        matrices={
            x: Matrix([[x, -x], [-y, 2 * x + y + 1]]),
            y: Matrix([[1 + y, -x], [-1 - y, x + 2 * y + 1]]),
        },
        validate=False,
    )


def symmetric_pi():
    return CMF(
        matrices={
            x: Matrix([[x, x * y], [1, 1 + 2 * x + y]]),
            y: Matrix([[y, x * y], [1, 1 + x + 2 * y]]),
        },
        validate=False,
    )


def zeta3() -> CMF:
    return CMF(
        matrices={
            x: Matrix(
                [
                    [0, -(x**3)],
                    [(x + 1) ** 3, x**3 + (x + 1) ** 3 + 2 * y * (y - 1) * (2 * x + 1)],
                ]
            ),
            y: Matrix(
                [
                    [-(x**3) + 2 * x**2 * y - 2 * x * y**2 + y**3, -(x**3)],
                    [x**3, x**3 + 2 * x**2 * y + 2 * x * y**2 + y**3],
                ]
            ),
        },
        validate=False,
    )


def var_root_cmf() -> CMF:
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
        matrices={x: Matrix([[0, b], [1, a]]), y: Matrix([[G, b], [1, F]])},
        validate=False,
    )


def cmf1() -> CMF:
    return FFbar(f=c0 + c1 * (x + y), fbar=c2 + c3 * (x - y))


def cmf2() -> CMF:
    return FFbar(
        f=(
            (2 * c1 + c2) * (c1 + c2)
            - c3 * c0
            - c3 * ((2 * c1 + c2) * (x + y) + (c1 + c2) * (2 * x + y))
            + c3**2 * (2 * x**2 + 2 * x * y + y**2)
        ),
        fbar=c3 * (c0 + c2 * x + c1 * y) - c3**2 * (2 * x**2 - 2 * x * y + y**2),
    )


def cmf3_1() -> CMF:
    return FFbar(
        f=-((c0 + c1 * (x + y)) * (c0 * (x + 2 * y) + c1 * (x**2 + x * y + y**2))),
        fbar=(c0 + c1 * (-x + y)) * (c0 * (x - 2 * y) - c1 * (x**2 - x * y + y**2)),
    )


def cmf3_2() -> CMF:
    return FFbar(
        f=-(x + y) * (c0**2 + 2 * c1**2 * (x**2 + x * y + y**2)),
        fbar=(x - y) * (c0**2 + 2 * c1**2 * (x**2 - x * y + y**2)),
    )


def cmf3_3() -> CMF:
    return FFbar(
        f=(x + y) * (c0**2 - c0 * c1 * (x - y) - 2 * c1**2 * (x**2 + x * y + y**2)),
        fbar=(c0 + c1 * (x - y)) * (3 * c0 * (x - y) + 2 * c1 * (x**2 - x * y + y**2)),
    )


def hypergeometric_derived_2F1() -> CMF:
    return CMF(
        matrices={
            a: Matrix(
                [[1 + 2 * a, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 4 * a + 2 * b + 4 * c]]
            ),
            b: Matrix(
                [[1 + 2 * b, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 2 * a + 4 * b + 4 * c]]
            ),
            c: Matrix(
                [
                    [-1 - 2 * c, (1 + 2 * a) * (1 + 2 * b)],
                    [1, 3 + 2 * a + 2 * b + 2 * c],
                ]
            ),
        },
        validate=False,
    )


def hypergeometric_derived_3F2() -> CMF:
    Sx = x0 + x1 + x2
    Sy = y0 + y1
    Tx = x0 * x1 + x0 * x2 + x1 * x2
    Ty = y0 * y1
    Px = x0 * x1 * x2
    M = Matrix(
        [
            [0, 0, Px / ((1 - z) * z)],
            [z, 1, ((Tx + Sx + 1) * z - Ty) / ((1 - z) * z)],
            [0, z, ((Sx + 1) * z + Sy + 1) / (1 - z)],
        ],
    )
    I3 = sp.eye(3)
    return CMF(
        matrices={
            x0: M / x0 + I3,
            x1: M / x1 + I3,
            x2: M / x2 + I3,
            y0: -M / (y0 + 1) + I3,
            y1: -M / (y1 + 1) + I3,
        },
        validate=False,
    )
