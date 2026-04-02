import pytest
import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.asymptotics import GrowthRate, PrecisionExhaustedError, Reducer
from ramanujantools.asymptotics.series_matrix import SeriesMatrix


def asymptotic_expressions(asymptotic_growth: list[GrowthRate]) -> list[sp.Expr]:
    return [g.as_expr(n) if g is not None else sp.S.Zero for g in asymptotic_growth]


def get_reducer(matrix: Matrix, precision: int = 5) -> Reducer:
    var = n
    degrees = [d for d in matrix.degrees(var) if d != -sp.oo]
    factorial_power = max(degrees) if degrees else 0

    normalized = matrix / (var**factorial_power)
    series = SeriesMatrix.from_matrix(normalized, var=n, p=1, precision=precision)

    return Reducer(
        series=series,
        factorial_power=factorial_power,
        precision=precision,
        p=1,
    )


def test_fibonacci():
    M = Matrix([[0, 1], [1, 1]])

    exprs = asymptotic_expressions(get_reducer(M).asymptotic_growth())

    expected_exprs = [
        (sp.Rational(1, 2) + sp.sqrt(5) / 2) ** n,
        (sp.Rational(1, 2) - sp.sqrt(5) / 2) ** n,
    ]

    assert [sp.simplify(e) for e in exprs] == expected_exprs


def test_tribonacci():
    R = (sp.sqrt(33) / 9 + sp.Rational(19, 27)) ** sp.Rational(1, 3)
    c1 = sp.Rational(-1, 2) - sp.sqrt(3) * sp.I / 2
    c2 = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2

    expected_bases = [
        sp.Rational(1, 3) + 4 / (9 * R) + R,
        sp.Rational(1, 3) + c1 * R + 4 / (9 * c1 * R),
        sp.Rational(1, 3) + 4 / (9 * c2 * R) + c2 * R,
    ]

    M = Matrix([[0, 0, 1], [1, 0, 1], [0, 1, 1]])

    growths = get_reducer(M).asymptotic_growth()
    assert len(growths) == 3

    actual_bases = [g.exp_base for g in growths]

    for expected, actual in zip(expected_bases, actual_bases):
        assert abs(sp.N(expected - actual, 50)) < 1e-40, (
            f"Expected {expected}, got {actual}"
        )


def test_exponential_separation():
    expected_lambda = Matrix.diag(4, 2)
    expected_D = Matrix.diag(5, 3)

    M_canonical = expected_lambda * (Matrix.eye(2) + expected_D / n)

    U = Matrix.eye(2) + Matrix([[1, -2], [3, 1]]) / n
    M = M_canonical.coboundary(U)

    exprs = asymptotic_expressions(get_reducer(M, precision=5).asymptotic_growth())

    expected_exprs = [4**n * n**5, 2**n * n**3]

    assert [sp.simplify(e) for e in exprs] == expected_exprs


def test_newton_polygon_separation():
    expected_canonical = Matrix([[4 * (1 + 1 / n) ** 3, 0], [0, 2 * (1 + 1 / n) ** 1]])
    U = Matrix([[1, n], [0, 1]])
    M = expected_canonical.coboundary(U)

    exprs = asymptotic_expressions(get_reducer(M, precision=5).asymptotic_growth())

    assert len(exprs) == 2

    assert 4**n in sp.simplify(exprs[0]).atoms(sp.Pow)
    assert 2**n in sp.simplify(exprs[1]).atoms(sp.Pow)


def test_ramified_scalar_peeling_no_block_degeneracy():
    """
    Triggers a Jordan block, hits the Identity Trap, uses Scalar Peeling
    to find distinct roots at M_1, and extracts them perfectly.
    """
    M = Matrix([[0, -(n - 1) / n], [1, 2]])
    exprs = asymptotic_expressions(
        get_reducer(M.transpose(), precision=4).asymptotic_growth()
    )

    expected_exprs = [
        sp.exp(-2 * sp.sqrt(n)) * n ** sp.Rational(-1, 4),
        sp.exp(2 * sp.sqrt(n)) * n ** sp.Rational(-1, 4),
    ]

    assert [sp.simplify(e) for e in exprs] == expected_exprs


@pytest.mark.parametrize(
    "U",
    [
        Matrix.eye(2),
        Matrix([[2, 0], [0, 5]]),
        Matrix([[0, 1], [1, 0]]),
        Matrix([[1, 1 / n], [0, 1]]),
        Matrix([[1, 0], [1 / (n**2), 1]]),
    ],
)
def test_gauge_invariance(U):
    M = Matrix([[0, -(n - 1) / n], [1, 2]])

    original_exprs = asymptotic_expressions(get_reducer(M).asymptotic_growth())
    transformed_exprs = asymptotic_expressions(
        get_reducer(M.coboundary(U)).asymptotic_growth()
    )

    assert [sp.simplify(e) for e in original_exprs] == [
        sp.simplify(e) for e in transformed_exprs
    ], f"Invariance failed for gauge U = {U}"


def test_nilpotent_ghost():
    m = Matrix([[0, 1], [0, 0]])
    precision = 3

    # Must explicitly trigger the Blindness Radar
    with pytest.raises(PrecisionExhaustedError):
        get_reducer(m.transpose(), precision=precision).reduce()


def test_row_nullity():
    """
    Because the reduction strictly multiplies invertible matrices, an entire row
    can only vanish if the final extraction process is starved or the input is invalid.
    We unit test the radar directly to ensure it guards the exit.
    """
    m = Matrix([[1, 0], [0, 1]])  # Dummy valid matrix
    reducer = get_reducer(m, precision=5)

    # Craft a physically impossible CFM where Variable 1 has completely vanished
    broken_cfm = [
        [GrowthRate(polynomial_degree=2), GrowthRate(polynomial_degree=3)],
        [GrowthRate(), GrowthRate()],
    ]

    with pytest.raises(PrecisionExhaustedError):
        reducer._check_cfm_validity(broken_cfm)


def test_input_trancation():
    """
    Tests if the strict boundary alarms catch an aggressive sub-diagonal shear starvation.
    The term n^-2 produces g=2. For a 5x5, max_shift = 8.
    At precision=3, this organically starves the matrix on iteration 1.
    """
    m = Matrix(
        [
            [1, 1, 0, 0, 0],
            [n**-2, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    )

    with pytest.raises(PrecisionExhaustedError):
        get_reducer(m, precision=3).reduce()


def test_ramification_exact_expressions():
    """
    Tests that a system requiring fractional powers successfully triggers
    ramification and extracts the sub-exponential roots.
    """
    M = Matrix([[0, 1], [1 / n, 0]])
    exprs = asymptotic_expressions(get_reducer(M, precision=4).asymptotic_growth())

    expected_exprs = [
        (-1) ** n * n ** sp.Rational(1, 4) / sp.sqrt(sp.factorial(n)),
        n ** sp.Rational(1, 4) / sp.sqrt(sp.factorial(n)),
    ]

    assert exprs == expected_exprs


def test_ramification_structural_mechanics():
    m = Matrix([[0, 1, 0], [0, 0, 1], [n**2, 0, 0]])
    # f(n) = n^2 * a_(n-3)

    precision = 10
    reducer = get_reducer(m.transpose(), precision=precision)
    grid = reducer.canonical_growth_matrix()

    # 1. Verify internal engine state mapped the branch correctly
    assert reducer.p == 3

    # 2. Strict Mathematical Validation (No string parsing needed anymore!)
    # We just inspect the strongly-typed GrowthRate objects in the grid.
    found_fractional_power = False

    for row in grid:
        for cell in row:
            if (
                isinstance(cell.polynomial_degree, sp.Rational)
                and cell.polynomial_degree.q == 3
            ):
                found_fractional_power = True
                break

    assert found_fractional_power, (
        "Failed to mathematically verify fractional ramification powers in the GrowthRate objects."
    )
