import sympy as sp
from sympy.abc import x, y, n

from ramanujantools import Matrix, Limit, simplify


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


def test_can_call_numerical_subs():
    m = Matrix([[x, 1], [y, 2]])

    # not enough parameters
    assert not m._can_call_numerical_subs({x: 1})

    # some parameters are not integer
    assert not m._can_call_numerical_subs({x: 1, y: y})
    assert not m._can_call_numerical_subs({x: 1, y: sp.Rational(2, 3)})

    # rational matrix
    assert not (m / x)._can_call_numerical_subs({x: 1, y: 1})

    assert m._can_call_numerical_subs({x: 17, y: 31})


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


def test_subs_numerical_equivalent_to_symbolic():
    m = Matrix([[x, x**2], [13 + x, -x]])
    substitutions = {x: 5}
    assert m.xreplace(substitutions) == m.numerical_subs(substitutions)


def test_as_polynomial():
    m = Matrix([[1, 1 / x], [0, 3 / (x**2 - x)]])
    polynomial_m = Matrix([[x * (x - 1), x - 1], [0, 3]])
    assert polynomial_m == m.as_polynomial()


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
    expected = U * m * U.inverse().subs({n: n + 1})
    assert expected == m.coboundary(U)


def test_coboundary_inverse():
    m = Matrix([[1, n, 2], [3, n**2, 5 * n], [n - 7, n**2 + 1, n - 3]])
    U = Matrix([[3, 1, n - 2], [5, n**2 - 3 * n + 1, 0], [11 * n + 2, 3, n - 19]])
    assert m == m.coboundary(U).coboundary(U.inverse())


def test_is_companion():
    assert not Matrix.eye(3).is_companion()
    m = Matrix([[0, 0, n**3 - 1], [1, 0, n**2 + 3], [0, 1, 2 * n]])
    assert m.is_companion()
    assert not (m + Matrix.eye(3)).is_companion()


def test_companion_coboundary():
    m = Matrix([[1, n, 2], [3, n**2, 5 * n], [n - 7, n**2 + 1, n - 3]])
    assert m.coboundary(m.companion_coboundary_matrix()).is_companion()


def test_select_inflation_factor():
    factors = {n, n + 1, n + 3, n + 4}
    assert Matrix.select_inflation_factor(factors, 0) == n
    assert Matrix.select_inflation_factor(factors, 1) == n + 1
    assert Matrix.select_inflation_factor(factors, 2) == n + 4


def test_normalize_companion():
    m = Matrix(
        [
            [0, 0, 1 / (n * (n - 1) * (n - 4))],
            [1, 0, 1 / ((n - 1) * (n - 2))],
            [0, 1, 1],
        ]
    )
    expected = m.inflate(n - 1).inflate(n).inflate(n - 4)
    assert expected.is_polynomial()
    # optimal = m.inflate(n).inflate(n-2) - for future optimization
    assert expected == m.normalize_companion()


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
    assert companion.is_polynomial()


def test_companion_coboundary_two_variables():
    m = Matrix([[1, x, 2 * y], [3, x * y, 5 * y], [x - 7, (x + y) ** 2 + 1, y - 3]])
    assert m.coboundary(m.companion_coboundary_matrix(x), x).is_companion()
    assert m.coboundary(m.companion_coboundary_matrix(y), y).is_companion()


def test_inflation_coboundary():
    c = x - 7
    expected = Matrix(
        [
            [c.subs({x: x - 2}) * c.subs({x: x - 1}), 0, 0],
            [0, c.subs({x: x - 1}), 0],
            [0, 0, 1],
        ]
    )
    U = Matrix.inflation_coboundary_matrix(3, x - 7, symbol=x)
    assert expected == U


def test_inflate():
    M = Matrix([[0, n**2], [1, n + 1]])
    expected = Matrix([[0, n**2 * (n - 1) * (n - 2)], [1, (n + 1) * (n - 1)]])
    assert expected == M.inflate(n - 1)


def test_canonize_companion():
    M = Matrix([[0, 0, n**3], [1, 0, n**2 + 1], [0, 1, n - 17]])
    assert M.canonize_companion() == M.deflate(M[-1])


def test_companion_equivalent():
    M = Matrix([[0, 0, n**3], [1, 0, n**2 + 1], [0, 1, n - 17]])
    M1 = M.inflate(n - 31).deflate((4 * n - 3) ** 2)
    M2 = M.inflate(n**5)
    assert M1.companion_equivalent(M2)


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
