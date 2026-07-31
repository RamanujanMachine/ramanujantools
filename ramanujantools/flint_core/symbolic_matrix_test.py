import sympy as sp
import pytest
from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.flint_core import flint_ctx, SymbolicMatrix


def flintify(matrix: Matrix, fmpz=True) -> SymbolicMatrix:
    ctx = flint_ctx(matrix.free_symbols, fmpz)
    return SymbolicMatrix.from_sympy(matrix, ctx)


def test_factor():
    matrix = Matrix(
        [
            [1, n**2 + n, n**2 - n + 5],
            [3 * n + 9, n**2 - 1, 1 / (n + 1)],
            [n**2 + 7, n - 2, (n + 1) * (n - 3)],
        ],
    )

    assert matrix.applyfunc(sp.factor) == flintify(matrix).factor()


def test_mul():
    m1 = Matrix([[0, n**2], [1, 1 / n]])
    m2 = Matrix([[3, n - 2], [n, 5]])

    assert flintify(m1 * m2) == flintify(m1) * flintify(m2)
    assert flintify(m2 * m1) == flintify(m2) * flintify(m1)


def test_mul_scalar():
    m = Matrix([[0, n**2], [1, 1 / n]])

    assert flintify(m * 17) == flintify(m) * 17
    assert flintify(m * 17) == 17 * flintify(m)


def test_inverse():
    matrix = Matrix([[n + 1, n], [1, 2 * n + 3]])
    symbolic = flintify(matrix)

    assert symbolic.inverse().to_rt(factor=True) == matrix.inverse()
    assert (symbolic * symbolic.inverse()).to_rt(factor=True) == Matrix.eye(2)


def test_inverse_rational_entries():
    matrix = Matrix([[1 / (n + 1), n], [1, 2]])
    symbolic = flintify(matrix)

    assert symbolic.inverse().to_rt(factor=True) == matrix.inverse()


def test_inverse_multivariable():
    x, y = sp.symbols("x y")
    matrix = Matrix([[x + y, x], [y, x + 2 * y]])

    assert flintify(matrix).inverse().to_rt(factor=True) == matrix.inverse()


def test_inverse_singular():
    symbolic = flintify(Matrix([[1, n], [n, n**2]]))

    with pytest.raises(ZeroDivisionError, match="Singular polynomial matrix"):
        symbolic.inverse()


def test_walk():
    matrix = Matrix(
        [
            [1, n**2 + n, n**2 - n + 5],
            [3 * n + 9, n**2 - 1, 1 / (n + 1)],
            [n**2 + 7, n - 2, (n + 1) * (n - 3)],
        ],
    )

    expected = (matrix * matrix.subs({n: n + 1}) * matrix.subs({n: n + 2})).factor()
    assert expected == flintify(matrix).walk({n: 1}, 3, {n: n}).factor()


def test_walk_rational():
    matrix = Matrix(
        [
            [1, n**2 + n, n**2 - n + 5],
            [3 * n + 9, n**2 - 1, 1 / (n + 1)],
            [n**2 + 7, n - 2, (n + 1) * (n - 3)],
        ],
    )

    expected = (
        matrix.subs({n: n / 2})
        * matrix.subs({n: n / 2 + 1})
        * matrix.subs({n: n / 2 + 2})
    ).factor()

    assert expected == flintify(matrix, fmpz=False).walk({n: 1}, 3, {n: n / 2}).factor()
