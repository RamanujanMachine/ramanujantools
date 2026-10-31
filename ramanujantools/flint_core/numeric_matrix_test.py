
import pytest
import sympy as sp
from sympy.abc import n, x, y

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


def test_walk_list():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    rt_walk_matrices = m.walk({n: 2}, list(range(1, 100)), {n: 3})
    numeric_walk_matrices = NumericMatrix.walk(
        m, Position({n: 2}), list(range(1, 100)), Position({n: 3})
    )
    for rt_walk_matrix, numeric_walk_matrix in zip(
        rt_walk_matrices, numeric_walk_matrices
    ):
        assert rt_walk_matrix == numeric_walk_matrix.to_rt()


def product_walk(matrix, trajectory, iterations, start):
    result = Matrix.eye(matrix.rows)
    for i in range(iterations):
        result *= matrix.subs({key: start[key] + i * trajectory[key] for key in start})
    return result


def test_walk_multivariate():
    matrix = Matrix([[x * y + 1, (x - y) / (x + y + 3)], [x**2 / 2, 1 / (x + 1)]])
    trajectory, start = Position({x: 2, y: -1}), Position({x: 3, y: 100})
    assert (
        product_walk(matrix, trajectory, 40, start)
        == NumericMatrix.walk(matrix, trajectory, 40, start).to_rt()
    )


def test_walk_rational_positions():
    matrix = Matrix([[n, 1], [1, 1 / n]])
    trajectory = Position({n: sp.Rational(2, 3)})
    start = Position({n: sp.Rational(1, 2)})
    assert (
        product_walk(matrix, trajectory, 10, start)
        == NumericMatrix.walk(matrix, trajectory, 10, start).to_rt()
    )


@pytest.mark.parametrize(
    "entry, singular_step",
    [
        (1 / n, 3),
        ((n**2 - 1) / (n - 1), 4),
        (n * (n + 1) / (n + 3) ** 2, 0),
        (1 / (2 + 1 / n), 3),  # inner denominator only
        (1 / (1 + 1 / n), 2),  # outer denominator
    ],
)
def test_walk_singular_step(entry, singular_step):
    matrix = Matrix([[entry, 1], [1, n]])
    trajectory, start = Position({n: 1}), Position({n: -3})
    assert (
        product_walk(matrix, trajectory, singular_step, start)
        == NumericMatrix.walk(matrix, trajectory, singular_step, start).to_rt()
    )
    with pytest.raises(ZeroDivisionError):
        NumericMatrix.walk(matrix, trajectory, singular_step + 1, start)


@pytest.mark.parametrize("entry", [sp.sqrt(n), sp.sin(n), 2**n])
def test_walk_unsupported_entry(entry):
    matrix = Matrix([[entry, 1], [1, n]])
    with pytest.raises(NotImplementedError):
        NumericMatrix.walk(matrix, Position({n: 1}), 10, Position({n: 1}))
