from sympy.abc import x, y

from ramanujan import Matrix, simplify, zero, inf


def test_gcd():
    a = 2 * 3 * 5
    b = 2 * 3 * 7
    c = 2 * 5 * 7
    d = 3 * 5 * 7
    m = Matrix([[a, b], [c, d]])
    m *= 11
    assert 11 == m.gcd()


def test_gcd_reduce():
    initial = Matrix([[2, 3], [5, 7]])
    gcd = 17
    m = gcd * initial
    assert m.gcd() == gcd
    assert m.reduce() == initial


def test_walk_0():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 0, y: 1}, 0, {x: x, y: y}) == Matrix.eye(2)


def test_walk_1():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 1, y: 0}, 1, {x: x, y: y}) == m


def test_walk_list():
    iterations = [1, 2, 3, 17, 29, 53, 99]
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 2, y: 3}, iterations, {x: 5, y: 7}) == [
        m.walk({x: 2, y: 3}, i, {x: 5, y: 7}) for i in iterations
    ]


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


def test_zero_vector():
    assert zero() == Matrix([0, 1])


def test_inf_vector():
    assert inf() == Matrix([1, 0])
