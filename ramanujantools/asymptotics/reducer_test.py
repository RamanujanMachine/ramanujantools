import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.asymptotics.reducer import Reducer


def test_constant_asymptotics_fibonacci():
    M = Matrix([[0, 1], [1, 1]])
    deg, Lambda, D = Reducer(M).reduce()
    assert deg == 0
    assert D == Matrix.zeros(2)
    assert Lambda == Matrix([[1 / 2 - sp.sqrt(5) / 2, 0], [0, 1 / 2 + sp.sqrt(5) / 2]])


def test_exponential_separation():
    # We want a solution that grows like: 2^n * n^3  and  4^n * n^5
    expected_lambda = Matrix.diag(2, 4)
    expected_D = Matrix.diag(3, 5)

    # The canonical M(n) for this is Lambda * (I + D/n)
    M_canonical = expected_lambda * (Matrix.eye(2) + expected_D / n)

    # A rational gauge to scramble it
    U = Matrix.eye(2) + Matrix([[1, -2], [3, 1]]) / n
    M = M_canonical.coboundary(U)

    # Run the Reducer
    reducer = Reducer(M, precision=5)
    fact_power, actual_lambda, actual_D = reducer.reduce()

    assert fact_power == 0
    assert actual_lambda == expected_lambda
    assert actual_D == expected_D


def test_newton_polygon_separation():
    n = sp.Symbol("n")

    expected_lambda = Matrix.diag(2, 4)
    expected_D = Matrix.diag(1, 3)

    expected_canonical = Matrix([[2 * (1 + 1 / n) ** 1, 0], [0, 4 * (1 + 1 / n) ** 3]])

    U = Matrix([[1, n], [0, 1]])

    m = expected_canonical.coboundary(U)

    reducer = Reducer(m, precision=5)
    fact_power, actual_lambda, actual_D = reducer.reduce()

    assert fact_power == 0  # The true system had no factorial growth
    assert actual_lambda == expected_lambda

    diff = expected_D - actual_D
    assert diff.is_diagonal(), (
        "D_calc should only differ from D_expected on the diagonal."
    )

    for i in range(diff.shape[0]):
        assert diff[i, i].is_integer, "Shift must be an integer."

    valuations = reducer.S_total.valuations()

    for i in range(reducer.dim):
        missing_powers = diff[i, i]
        receipt_powers = valuations[i, i]
        assert receipt_powers == missing_powers, (
            f"Conservation of Growth violated at column {i}! "
            f"D shifted by {missing_powers}, but S_total recorded a shear of {receipt_powers}."
        )
