import pytest
import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.asymptotics import (
    GrowthRate,
    EigenvalueBlindnessError,
    RowNullityError,
    ShearOverflowError,
    Reducer,
)


def test_fibonacci():
    M = Matrix([[0, 1], [1, 1]])

    exprs = Reducer.from_matrix(M).asymptotic_expressions()

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

    growths = Reducer.from_matrix(M).asymptotic_growth()
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

    exprs = Reducer.from_matrix(M, precision=5).asymptotic_expressions()

    expected_exprs = [4**n * n**5, 2**n * n**3]

    assert [sp.simplify(e) for e in exprs] == expected_exprs


def test_newton_polygon_separation():
    expected_canonical = Matrix([[4 * (1 + 1 / n) ** 3, 0], [0, 2 * (1 + 1 / n) ** 1]])
    U = Matrix([[1, n], [0, 1]])
    M = expected_canonical.coboundary(U)

    exprs = Reducer.from_matrix(M, precision=5).asymptotic_expressions()

    assert len(exprs) == 2

    assert 4**n in sp.simplify(exprs[0]).atoms(sp.Pow)
    assert 2**n in sp.simplify(exprs[1]).atoms(sp.Pow)


def test_ramified_scalar_peeling_no_block_degeneracy():
    """
    Triggers a Jordan block, hits the Identity Trap, uses Scalar Peeling
    to find distinct roots at M_1, and extracts them perfectly.
    """
    M = Matrix([[0, -(n - 1) / n], [1, 2]])
    exprs = Reducer.from_matrix(M.transpose(), precision=4).asymptotic_expressions()

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

    original_exprs = Reducer.from_matrix(M).asymptotic_expressions()
    transformed_exprs = Reducer.from_matrix(M.coboundary(U)).asymptotic_expressions()

    assert [sp.simplify(e) for e in original_exprs] == [
        sp.simplify(e) for e in transformed_exprs
    ], f"Invariance failed for gauge U = {U}"


def test_nilpotent_ghost():
    m = Matrix([[0, 1], [0, 0]])
    precision = 3

    # Must explicitly trigger the Blindness Radar
    with pytest.raises(EigenvalueBlindnessError) as e:
        reducer = Reducer.from_matrix(m.transpose(), precision=precision)
        reducer.canonical_fundamental_matrix()

    # Strictly validate the mathematical jump requested
    assert e.value.required_precision == precision + 2


def test_row_nullity():
    """
    Because the reduction strictly multiplies invertible matrices, an entire row
    can only vanish if the final extraction process is starved or the input is invalid.
    We unit test the radar directly to ensure it guards the exit.
    """
    m = Matrix([[1, 0], [0, 1]])  # Dummy valid matrix
    reducer = Reducer.from_matrix(m, precision=5)

    # Craft a physically impossible CFM where Variable 1 has completely vanished
    broken_cfm = [
        [GrowthRate(polynomial_degree=2), GrowthRate(polynomial_degree=3)],
        [GrowthRate(), GrowthRate()],
    ]

    with pytest.raises(RowNullityError) as e:
        reducer._check_cfm_validity(broken_cfm)

    assert e.value.required_precision == 7  # precision + dim


def test_shear_overflow():
    """
    Tests if the Overflow Radar catches an aggressive sub-diagonal shear.
    The term n^-2 produces g=2. For a 3x3, max_shift = 4.
    At precision=3, this organically overflows on iteration 1.
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

    with pytest.raises(ShearOverflowError) as e:
        from ramanujantools.asymptotics.reducer import Reducer

        Reducer.from_matrix(m, precision=3).canonical_fundamental_matrix()

    assert e.value.required_precision >= 5


def test_ramification_exact_expressions():
    """
    Tests that a system requiring fractional powers successfully triggers
    ramification and extracts the sub-exponential roots.
    """
    M = Matrix([[0, 1], [1 / n, 0]])
    exprs = Reducer.from_matrix(M, precision=4).asymptotic_expressions()
    print(exprs)

    expected_exprs = [
        (-1) ** n * n ** sp.Rational(1, 4) / sp.sqrt(sp.factorial(n)),
        n ** sp.Rational(1, 4) / sp.sqrt(sp.factorial(n)),
    ]

    assert exprs == expected_exprs


def test_ramification_structural_mechanics():
    m = Matrix([[0, 1, 0], [0, 0, 1], [n**2, 0, 0]])
    # f(n) = n^2 * a_(n-3)

    precision = 10
    reducer = Reducer.from_matrix(m.transpose(), precision=precision)
    cfm = reducer.canonical_fundamental_matrix()

    # 1. Verify internal engine state mapped the branch correctly
    assert reducer.p == 3

    # 2. Strict Mathematical Validation (No string checks)
    # We traverse the SymPy expression tree looking for Rational exponents.
    # The ramification must mathematically produce an exponent with a denominator of 3.
    found_fractional_power = False

    for element in cfm:
        # Extract all base-exponent pairs (e.g., n**(1/3) -> base=n, exp=1/3)
        for power in element.as_expr(n).atoms(sp.Pow):
            base, exp = power.as_base_exp()
            if base == n and isinstance(exp, sp.Rational) and exp.q == 3:
                found_fractional_power = True
                break

    assert found_fractional_power, (
        "Failed to mathematically verify fractional ramification powers."
    )


def test_euler_trajectory():
    p3 = -8 * n - 11
    p2 = 24 * n**3 + 105 * n**2 + 124 * n + 25
    p1 = -((n + 2) ** 3) * (24 * n**2 + 97 * n + 94)
    p0 = (n + 1) ** 4 * (n + 2) ** 2 * (8 * n + 19)

    M = Matrix([[0, 0, -p0 / p3], [1, 0, -p1 / p3], [0, 1, -p2 / p3]])

    expected = [
        n ** (sp.Rational(22, 3))
        * sp.exp(-3 * n ** (sp.Rational(2, 3)) + n ** (sp.Rational(1, 3)))
        * sp.factorial(n) ** 2,
        n ** (sp.Rational(22, 3))
        * sp.exp(
            -(n ** (sp.Rational(1, 3)))
            * (6 * sp.I * n ** (sp.Rational(1, 3)) + sp.sqrt(3) + sp.I)
            / (sp.sqrt(3) - sp.I)
        )
        * sp.factorial(n) ** 2,
        n ** (sp.Rational(22, 3))
        * sp.exp(
            n ** (sp.Rational(1, 3))
            * (
                3 * sp.sqrt(3) * n ** (sp.Rational(1, 3))
                + 3 * sp.I * n ** (sp.Rational(1, 3))
                + 2 * sp.I
            )
            / (sp.sqrt(3) - sp.I)
        )
        * sp.factorial(n) ** 2,
    ]
    reducer = Reducer.from_matrix(M.transpose(), precision=9)
    actual = reducer.asymptotic_expressions()
    assert expected == actual
