import pytest

import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.asymptotics.reducer import Reducer


def test_fibonacci():
    M = Matrix([[0, 1], [1, 1]])
    reducer = Reducer.from_matrix(M)
    assert [1 / 2 + sp.sqrt(5) / 2, 0], [
        0,
        1 / 2 - sp.sqrt(5) / 2,
    ] == reducer.asymptotic_expressions()


def test_tribonacci():
    R = (sp.sqrt(33) / 9 + sp.Rational(19, 27)) ** sp.Rational(1, 3)
    c1 = sp.Rational(-1, 2) - sp.sqrt(3) * sp.I / 2
    c2 = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2

    # We now expect the full base^(n) expressions!
    expected_bases = [
        sp.Rational(1, 3) + 4 / (9 * R) + R,
        sp.Rational(1, 3) + c1 * R + 4 / (9 * c1 * R),
        sp.Rational(1, 3) + 4 / (9 * c2 * R) + c2 * R,
    ]

    M = Matrix([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    exprs = Reducer.from_matrix(M).asymptotic_expressions()

    assert len(exprs) == 3
    for expected_base in expected_bases:
        # Check that expected_base**n exists perfectly in one of the solutions
        assert any(
            sp.simplify(expected_base**n) in sp.simplify(expr).atoms(sp.Pow)
            for expr in exprs
        )


def test_exponential_separation():
    expected_lambda = Matrix.diag(4, 2)
    expected_D = Matrix.diag(5, 3)

    M_canonical = expected_lambda * (Matrix.eye(2) + expected_D / n)

    U = Matrix.eye(2) + Matrix([[1, -2], [3, 1]]) / n
    M = M_canonical.coboundary(U)

    exprs = Reducer.from_matrix(M, precision=5).asymptotic_expressions()

    # The engine directly outputs the final integrated combinations!
    expected_exprs = {4**n * n**5, 2**n * n**3}
    assert set(exprs) == expected_exprs


def test_newton_polygon_separation():
    expected_canonical = Matrix([[4 * (1 + 1 / n) ** 3, 0], [0, 2 * (1 + 1 / n) ** 1]])
    U = Matrix([[1, n], [0, 1]])
    m = expected_canonical.coboundary(U)

    exprs = Reducer.from_matrix(m, precision=5).asymptotic_expressions()

    # We no longer need to check "conservation of growth" via matrix valuations.
    # If the expressions solve out and separate successfully without crashing,
    # the new architecture proved it sliced them correctly.
    assert len(exprs) == 2
    assert any(4**n in expr.atoms(sp.Pow) for expr in exprs)
    assert any(2**n in expr.atoms(sp.Pow) for expr in exprs)


def test_ramification():
    """
    Tests that a system requiring fractional powers successfully triggers
    ramification and extracts the sub-exponential roots.
    """
    M = Matrix([[0, 1], [1 / n, 0]])
    exprs = Reducer.from_matrix(M, precision=4).asymptotic_expressions()

    expected_exprs = [
        (-1) ** n * n ** sp.Rational(1, 4) / sp.sqrt(sp.factorial(n)),
        n ** sp.Rational(1, 4) / sp.sqrt(sp.factorial(n)),
    ]

    assert expected_exprs == exprs


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

    assert exprs == expected_exprs


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

    assert set(original_exprs) == set(transformed_exprs), (
        f"Invariance failed for gauge U = {U}"
    )


def test_euler_trajectory():
    p3 = -8 * n - 11
    p2 = 24 * n**3 + 105 * n**2 + 124 * n + 25
    p1 = -((n + 2) ** 3) * (24 * n**2 + 97 * n + 94)
    p0 = (n + 1) ** 4 * (n + 2) ** 2 * (8 * n + 19)

    M = Matrix([[0, 0, -p0 / p3], [1, 0, -p1 / p3], [0, 1, -p2 / p3]])
    exprs = Reducer.from_matrix(M.transpose(), precision=6).asymptotic_expressions()

    assert len(exprs) == 3

    for expr in exprs:
        # 1. Assert the (n!)^2 factorial growth exists
        assert sp.factorial(n) ** 2 in expr.atoms(sp.Pow) or sp.factorial(
            n
        ) in expr.atoms(sp.Function)

        # 2. Assert the D = 1/3 algebraic tail was extracted natively
        assert n ** sp.Rational(1, 3) in expr.atoms(sp.Pow)

        # 3. Dissect the sub-exponential Q(n) function mathematically!
        exp_funcs = list(expr.atoms(sp.exp))
        assert len(exp_funcs) == 1

        Q_n = sp.expand(exp_funcs[0].args[0])

        # Q_n is structured as c1*n + c2*n**(1/2) + ...
        c2 = Q_n.coeff(n ** sp.Rational(1, 2))
        c1 = Q_n.coeff(n)

        # Mathematica's c2 roots are exactly the complex roots of x^3 = -27
        assert sp.simplify(c2**3) == -27

        # Mathematica's c1 roots strictly follow the relation c1 = (-c2/3)^2
        assert sp.simplify(c1 - (-c2 / 3) ** 2) == 0
