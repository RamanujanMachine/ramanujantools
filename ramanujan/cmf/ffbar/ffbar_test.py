from sympy.abc import x, y

from ramanujan import SquareMatrix
from ramanujan.cmf import CMF, known_cmfs
from ramanujan.cmf.known_cmfs import c0, c1, c2, c3
from ramanujan.cmf.ffbar import FFbar


def test_linear_condition():
    assert FFbar.linear_condition(x + y, x - y) == 0
    assert FFbar.linear_condition(2**x - 3**y, 2**x + 3**y) != 0


def test_quadratic_condition():
    assert FFbar.quadratic_condition(x**2 + x * y + y**2, x - y) == 0
    assert FFbar.quadratic_condition(x**2 + x * y + y**2, x - y + 1) != 0


def test_ffbar_construction():
    c0, c1, c2,
    assert known_cmfs.cmf1() == CMF(
        matrices={
            x: SquareMatrix(
                [
                    [0, -c0 * c2 + (c0 + c1 * x) * (c2 + c3 * x)],
                    [1, c0 + c1 * (x + y) - c2 - c3 * (x - y + 1)],
                ]
            ),
            y: SquareMatrix(
                [
                    [c2 + c3 * (x - y), -c0 * c2 + (c0 + c1 * x) * (c2 + c3 * x)],
                    [1, c0 + c1 * (x + y)],
                ]
            ),
        }
    )
