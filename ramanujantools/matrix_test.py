from pytest import approx

import sympy as sp
from sympy.abc import x, y, n

from ramanujantools import Matrix, Limit, simplify
from ramanujantools.pcf import PCF
from ramanujantools.cmf import pFq


def test_is_square():
    assert Matrix([[1, 2], [3, 4]]).is_square()
    assert not Matrix([1, 2, 3, 4]).is_square()


def test_gcd():
    a = 2 * 3 * 5
    b = 2 * 3 * 7
    c = 2 * 5 * 7
    d = 3 * 5 * 7
    m = Matrix([[a, b], [c, d]])
    m *= 11
    assert 11 == m.gcd


def test_reduce():
    initial = Matrix([[2, 3], [5, 7]])
    gcd = sp.Rational(17, 13)
    m = gcd * initial
    assert m.gcd == gcd
    assert m.reduce() == initial


def test_is_numeric_walk():
    m = Matrix([[x, 1], [y, 2]])

    assert m._is_numeric_walk({x: 1, y: 1}, {x: 1, y: 1})
    assert m._is_numeric_walk(
        {x: 1, y: 1}, {x: sp.Rational(1, 2), y: sp.Rational(1, 2)}
    )

    # x remains a symbol
    assert not m._is_numeric_walk({x: 1, y: 1}, {x: x, y: 1})

    # rational coefficients of symbolic polynomials are supported
    assert not m._is_numeric_walk({x: 1, y: 1}, {x: x / 2, y: 1})

    # substituting a different symbol causes symbolic flint calculation
    assert not m._is_numeric_walk({x: 1, y: 1}, {x: n - 1, y: n**2})


def test_subs_degenerated():
    m = Matrix([[x, 1], [y, 2]])
    assert m == m.subs({x: x})
    assert m == m.subs({y: y})
    assert m == m.subs({x: x, y: y})


def test_subs_numerical():
    m = Matrix([[x, x**2], [13 + x, -x]])
    substitutions = {x: 5}
    assert Matrix([[5, 25], [18, -5]]) == m.subs(substitutions)


def test_subs_symbolic():
    m = Matrix([[x, x**2, 13 + x, -x]])
    expr = y**2 + x - 3
    assert Matrix([[expr, (expr) ** 2, 13 + expr, -expr]]) == m.subs({x: expr})


def test_as_polynomial():
    m = Matrix([[1, 1 / x], [0, 3 / (x**2 - x)]])
    polynomial_m = Matrix([[x * (x - 1), x - 1], [0, 3]])
    assert polynomial_m == m.as_polynomial()


def test_factor_symbolic():
    matrix = Matrix(
        [
            [x**3 + 3 * x**2 + 3 * x + 1, x**100 - x**90 + x**80],
            [(x + 1) ** 2, 3 / (x**2 - x)],
        ]
    )
    assert matrix.applyfunc(sp.factor) == matrix.factor()


def test_factor_numeric():
    matrix = Matrix(
        [[0, -1, 2, -3], [-4, 5, -6, 7], [-8, 9, 10, -11], [12, -13, -14, 15]]
    )
    assert matrix.applyfunc(sp.factor) == matrix.factor()


def test_inverse():
    a = 5
    b = 2
    c = 3
    d = 7
    m = Matrix([[a, b], [c, d]])
    expected = Matrix([[d, -b], [-c, a]]) / (a * d - b * c)
    assert expected == m.inverse()


def test_singular_points_nonvariable():
    m = Matrix([[1, 2], [3, 4]])
    assert len(m.singular_points()) == 0


def test_singular_points_single_variable():
    m = Matrix([[1, 0], [1, (x - 1) * (x - 3)]])
    assert m.singular_points() == [{x: 1}, {x: 3}]


def test_singular_points_multi_variable():
    m = Matrix([[1, x], [1, y]])
    assert m.singular_points() == [{x: y}]


def test_coboundary():
    m = Matrix([[1, n, 2], [3, n**2, 5 * n], [n - 7, n**2 + 1, n - 3]])
    U = Matrix([[3, 1, n - 2], [5, n**2 - 3 * n + 1, 0], [11 * n + 2, 3, n - 19]])
    expected = U.inverse() * m * U.subs({n: n + 1})
    assert expected == m.coboundary(U)


def test_coboundary_inverse():
    m = Matrix([[1, n, 2], [3, n**2, 5 * n], [n - 7, n**2 + 1, n - 3]])
    U = Matrix([[3, 1, n - 2], [5, n**2 - 3 * n + 1, 0], [11 * n + 2, 3, n - 19]])
    assert m == m.coboundary(U).coboundary(U.inverse())


def test_companion_form():
    expected = Matrix([[0, 0, n], [1, 0, 17], [0, 1, n**2]])
    assert expected == Matrix.companion_form([n, 17, n**2])


def test_is_companion():
    assert not Matrix.eye(3).is_companion()
    m = Matrix([[0, 0, n**3 - 1], [1, 0, n**2 + 3], [0, 1, 2 * n]])
    assert m.is_companion()
    assert not (m + Matrix.eye(3)).is_companion()


def test_companion_coboundary():
    m = Matrix([[1, n, 2], [3, n**2, 5 * n], [n - 7, n**2 + 1, n - 3]])
    assert m.coboundary(m.companion_coboundary_matrix()).is_companion()


def test_as_companion():
    m = Matrix(
        [
            [1, -1, -n * (1 - 1 / n) - n - 1],
            [2 / n, 1 - 2 / n, -2 + (-2 * n - 1) / n + (-n - 1) / n + 2 / n],
            [
                n ** (-2),
                (1 - 1 / n) / n + 1 / n,
                (1 - 1 / n) ** 2 + (-2 * n - 1) / n**2,
            ],
        ]
    )

    companion = m.as_companion()
    assert companion.is_companion()


def test_as_companion_smaller_recursion():
    # This matrix is derived from pFq(4, 3, 1) with
    # start = {x0: 0, x1: 0, x2: 1, x3: 1, y0: 1, y1: 1, y2: 1},
    # trajectory = {x0: -1, x1: -1, x2: 1, x3: 1, y0: 0, y1: 0, y2: 0}
    m = Matrix(
        [
            [(17 * n - 12) / n, 12 * n - 8, 4 * n * (2 * n - 1)],
            [(24 * n - 24) / n**2, (17 * n - 16) / n, 12 * n - 8],
            [-12 / n**3, -8 / n**2, (n - 4) / n],
        ]
    )
    companion = m.as_companion()
    assert companion.is_companion()
    assert 2 == companion.rows
    # This is Apery's PCF
    assert PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6)) == PCF(companion)


def test_companion_coboundary_two_variables():
    m = Matrix([[1, x, 2 * y], [3, x * y, 5 * y], [x - 7, (x + y) ** 2 + 1, y - 3]])
    assert m.coboundary(m.companion_coboundary_matrix(x), x).is_companion()
    assert m.coboundary(m.companion_coboundary_matrix(y), y).is_companion()


def test_walk_0():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 0, y: 1}, 0, {x: x, y: y}) == Matrix.eye(2)


def test_walk_1():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 1, y: 0}, 1, {x: x, y: y}) == m


def test_walk_list():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29, 53, 99]
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk(trajectory, iterations, start) == [
        m.walk(trajectory, i, start) for i in iterations
    ]


def test_walk_start_single_variable():
    iterations = [1, 2, 3, 4]
    m = Matrix([[0, x**2], [1, x + 1]])
    expected = m.walk({x: 1}, sum(iterations), {x: 1})
    actual = Matrix.eye(2)
    for i in range(len(iterations)):
        actual *= m.walk({x: 1}, iterations[i], {x: 1 + sum(iterations[0:i])})
    assert expected == actual


def test_walk_start_multi_variable():
    iterations = [1, 2, 3, 4]
    m = Matrix([[0, x**2], [1, y + 1]])
    starting_point = {x: 2, y: 3}
    trajectory = {x: 5, y: 7}
    expected = m.walk(trajectory, sum(iterations), starting_point)
    actual = Matrix.eye(2)
    for i in range(len(iterations)):
        actual *= m.walk(
            trajectory,
            iterations[i],
            {
                x: starting_point[x] + sum(iterations[0:i]) * trajectory[x],
                y: starting_point[y] + sum(iterations[0:i]) * trajectory[y],
            },
        )
    assert expected == actual


def test_walk_axis():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 1, y: 0}, 3, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 2, y: 1}) * m({x: 3, y: 1})
    )
    assert simplify(m.walk({x: 0, y: 1}, 5, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1})
        * m({x: 1, y: 2})
        * m({x: 1, y: 3})
        * m({x: 1, y: 4})
        * m({x: 1, y: 5})
    )


def test_walk_diagonal():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 1, y: 1}, 4, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 2, y: 2}) * m({x: 3, y: 3}) * m({x: 4, y: 4})
    )
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 4, y: 3}) * m({x: 7, y: 5})
    )


def test_walk_different_start():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 5, y: 7})) == simplify(
        m({x: 5, y: 7}) * m({x: 8, y: 9}) * m({x: 11, y: 11})
    )


def test_walk_equivalent_to_limit():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = 17
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    m_previous, m_last = m.walk(trajectory, [iterations - 1, iterations], start)
    assert m.limit(trajectory, iterations, start) == Limit(m_last, m_previous)


def test_limit_list():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29, 53, 99]
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    limits = m.limit(trajectory, iterations, start)
    for depth_index in range(len(iterations)):
        assert limits[depth_index] == m.limit(
            trajectory, iterations[depth_index], start
        )


def test_limit_initial_values():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = 10
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    initial_values = Matrix([[2, 3, 5], [7, 11, 13]])
    limit = m.limit(trajectory, iterations, start, initial_values)
    assert initial_values == limit.initial_values
    assert limit != m.limit(trajectory, iterations, start)


def test_charpoly():
    m = Matrix([[0, -(n**2)], [1, (3 * n + 1)]])
    assert sp.Matrix(m).charpoly() == m.charpoly(poincare=False)


def test_poincare_poly():
    poly = sp.PurePoly(2 * x**3 + x**2 + x * (4 * n**2 + 2) + (n**3), x)
    expected = sp.PurePoly(2 * x**3 + 4 * x + 1, x)
    assert expected == Matrix.poincare_poly(poly)


def test_poincare_poly_degenerated():
    poly = sp.PurePoly(2 * x**3 + x**2 + x * (4 * n**2 + 2) + (n**4), x)
    expected = sp.PurePoly(2 * x**3, x)
    assert expected == Matrix.poincare_poly(poly)


def test_poincare_poly_aptekarev():
    poly = sp.PurePoly(
        x**3
        + (-256 * n**3 - 528 * n**2 - 352 * n - 73) / (16 * n + 1) * x**2
        + (2048 * n**4 + 2816 * n**3 - 632 * n**2 - 2114 * n - 765)
        / (256 * n**2 - 224 * n - 15)
        * x
        + (-16 * n**3 - 17 * n**2) / (16 * n - 15),
        x,
    )
    expected = sp.PurePoly(x**3 - 16 * x**2, x)
    assert expected == Matrix.poincare_poly(poly)


def test_poincare_poly_constant():
    poly = sp.PurePoly(5 * x**3 + 2 * x**2 + x - 7, x)
    assert poly == Matrix.poincare_poly(poly)


def test_errors():
    x0, x1 = sp.symbols("x:2")
    (y0,) = sp.symbols("y:1")
    trajectory = {x0: 1, x1: 1, y0: 1}
    start = trajectory
    m = pFq(2, 1, -1).trajectory_matrix(trajectory, start)
    lambdas = m.sorted_eigenvals()
    assert sp.log(abs(lambdas[0]).evalf() / abs(lambdas[1]).evalf()) == approx(
        m.errors()[0]
    )


def test_gcd_slope():
    x0, x1 = sp.symbols("x:2")
    (y0,) = sp.symbols("y:1")
    trajectory = {x0: 1, x1: 1, y0: 1}
    start = {x0: 0, x1: 0, y0: 0}
    m = pFq(2, 1, -1).trajectory_matrix(trajectory, start)
    assert m.gcd_slope(20) == approx(1.3962425331281643)
    assert m.gcd_slope(40) == approx(1.5535146470266243)
    assert m.gcd_slope(100) == approx(1.5895449981095848)


def test_kamidelta_2f1():
    x0, x1 = sp.symbols("x:2")
    (y0,) = sp.symbols("y:1")
    trajectory = {x0: 1, x1: 2, y0: 3}
    start = trajectory
    m = pFq(2, 1, -1).trajectory_matrix(trajectory, start)
    actual = m.kamidelta()[0]
    l1, l2 = m.limit({n: 1}, [100, 200], {n: 1})
    expected = l1.delta(l2.as_float())
    assert actual == approx(expected, abs=1e-1)  # at most 0.1 error


def test_kamidelta_2f2():
    x0, x1 = sp.symbols("x:2")
    y0, y1 = sp.symbols("y:2")
    trajectory = {x0: 1, x1: 1, y0: 0, y1: -1}
    start = {x0: 1, x1: 1, y0: -1, y1: -1}
    m = pFq(2, 2, -1).trajectory_matrix(trajectory, start)
    actual = m.kamidelta()[0]
    l1, l2 = m.limit({n: 1}, [100, 200], {n: 1})
    expected = l1.delta(l2.as_float())
    assert actual == approx(expected, abs=1e-1)  # at most 0.1 error


def test_kamidelta_3f2():
    x0, x1, x2 = sp.symbols("x:3")
    y0, y1 = sp.symbols("y:2")
    trajectory = {x0: -1, x1: 1, x2: -2, y0: 2, y1: -3}
    start = trajectory
    m = pFq(3, 2, -1).trajectory_matrix(trajectory, start)
    actual = m.kamidelta(depth=100)[0]
    l1, l2 = m.limit({n: 1}, [100, 200], {n: 1})
    expected = l1.delta(l2.as_float())
    assert actual == approx(expected, abs=1e-1)  # at most 0.1 error
