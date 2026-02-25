import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.asymptotics.reducer import Reducer


def test_fibonacci():
    M = Matrix([[0, 1], [1, 1]])
    reducer = Reducer(M)
    deg, Lambda, D = reducer.reduce()
    assert deg == 0
    assert D == Matrix.zeros(2)
    assert Lambda == Matrix([[1 / 2 + sp.sqrt(5) / 2, 0], [0, 1 / 2 - sp.sqrt(5) / 2]])
    assert [
        Lambda[0, 0] ** n,
        Lambda[1, 1] ** n,
    ] == reducer.get_asymptotic_expressions()


def test_tribonacci():
    R = (sp.sqrt(33) / 9 + sp.Rational(19, 27)) ** sp.Rational(1, 3)

    c1 = sp.Rational(-1, 2) - sp.sqrt(3) * sp.I / 2
    c2 = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2

    expected_lambda = Matrix(
        [
            [sp.Rational(1, 3) + 4 / (9 * R) + R, 0, 0],
            [0, sp.Rational(1, 3) + c1 * R + 4 / (9 * c1 * R), 0],
            [0, 0, sp.Rational(1, 3) + 4 / (9 * c2 * R) + c2 * R],
        ]
    )

    M = Matrix([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    deg, Lambda, D = Reducer(M).reduce()
    assert deg == 0
    assert D == Matrix.zeros(3)
    assert expected_lambda.simplify() == Lambda.simplify()


def test_exponential_separation():
    # We want a solution that grows like: 2^n * n^3  and  4^n * n^5
    expected_lambda = Matrix.diag(4, 2)
    expected_D = Matrix.diag(5, 3)

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

    expected_lambda = Matrix.diag(4, 2)
    expected_D = Matrix.diag(3, 1)

    expected_canonical = Matrix([[4 * (1 + 1 / n) ** 3, 0], [0, 2 * (1 + 1 / n) ** 1]])

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


def test_ramification():
    """
    Tests that a system requiring fractional powers (like the Airy equation)
    successfully triggers Phase 4, ramifies the series, and extracts
    the fractional exponential roots.
    """
    n = sp.Symbol("n")

    # M(n) = [[0, 1], [1/n, 0]]
    # True eigenvalues are +n^{-1/2} and -n^{-1/2}
    M = Matrix([[0, 1], [1 / n, 0]])

    # Run the Reducer
    reducer = Reducer(M, precision=4)
    fact_power, Lambda, D = reducer.reduce()

    assert reducer.p == 2

    actual_eigenvalues = set(Lambda.diagonal())
    expected_eigenvalues = {sp.S(1), sp.S(-1)}

    assert actual_eigenvalues == expected_eigenvalues


def test_ramified_scalar_peeling_no_block_degeneracy():
    """
    Triggers a Jordan block, shears to p=2, hits the Identity Trap,
    uses Scalar Peeling to find distinct roots at M_1 (+1, -1),
    and solves the system WITHOUT fracturing into a block degeneracy!

    Recurrence: n*f_{n+2} - 2n*f_{n+1} + (n - 1)*f_n = 0
    """
    n = sp.Symbol("n")

    # Standard companion matrix for f_{n+2} = 2*f_{n+1} - ((n-1)/n)*f_n
    M = Matrix([[0, -(n - 1) / n], [1, 2]])

    # Transposed to match the established convention
    reducer = Reducer(M.transpose(), precision=4)
    deg, Lambda, D = reducer.reduce()

    # 1. No factorial growth (degree of all polynomial coeffs is 1)
    assert deg == 0

    # 2. Ramification perfectly detected (Newton Polygon slope 1/2)
    assert reducer.p == 2

    # 3. Canonical form is perfectly diagonalized! No cross-talk left.
    assert Lambda.is_diagonal()
    assert D.is_diagonal()


def test_euler_trajectory():
    p3 = -8 * n - 11
    p2 = 24 * n**3 + 105 * n**2 + 124 * n + 25
    p1 = -((n + 2) ** 3) * (24 * n**2 + 97 * n + 94)
    p0 = (n + 1) ** 4 * (n + 2) ** 2 * (8 * n + 19)

    M = Matrix([[0, 0, -p0 / p3], [1, 0, -p1 / p3], [0, 1, -p2 / p3]])

    reducer = Reducer(M.transpose(), precision=6)

    deg, Lambda, D = reducer.reduce()

    assert deg == 2
    assert reducer.p == 3

    M1 = reducer.M.coeffs[1]
    M2 = reducer.M.coeffs[2]
    M3 = reducer.M.coeffs[3]

    for i in range(3):
        l1 = M1[i, i]
        l2 = M2[i, i]
        l3 = M3[i, i]

        # Translate Difference Matrices to Scalar Exponents
        c2 = sp.Rational(3, 2) * l1
        c1 = 3 * (l2 - sp.Rational(1, 2) * l1**2)
        D_raw = l3 - l1 * l2 + sp.Rational(1, 3) * l1**3

        # Apply the Stirling correction to match Mathematica's format
        # (n!)^2 introduces a +1 to the polynomial power.
        D_math = sp.simplify(D_raw + sp.S.One)

        # Prove Equivalence to Mathematica!
        # Mathematica's c2 roots are exactly the complex roots of x^3 = -27
        assert sp.simplify(c2**3) == -27

        # Mathematica's c1 roots strictly follow the relation c1 = (-c2/3)^2
        assert sp.simplify(c1 - (-c2 / 3) ** 2) == 0

        # Mathematica's algebraic tail is exactly 1/3 for all solutions
        assert D_math == sp.Rational(1, 3)
