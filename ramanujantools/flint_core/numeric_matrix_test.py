from sympy.abc import n

from ramanujantools import Matrix, Position
from ramanujantools.flint_core import NumericMatrix


def test_conversion():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    for i in range(1, 10):
        assert m.subs({n: i}) == NumericMatrix.lambda_from_rt(m)({n: i}).to_rt()


def test_walk():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    assert (
        m.walk({n: 2}, 100, {n: 3})
        == NumericMatrix.walk(m, Position({n: 2}), 100, Position({n: 3})).to_rt()
    )
