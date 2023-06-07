from pytest import approx
from sympy.abc import n, x, y

from ramanujan import Position, Matrix, simplify


def test_position_iadd():
    a = Position({x: 57, y: 29})
    b = Position({x: 82, y: 31})
    a_b = Position({x: a[x] + b[x], y: a[y] + b[y]})
    a += b
    assert a == a_b


def test_limit():
    a, b, c, d = (8, 2, 19, 5)
    m = Matrix([[a, b], [c, d]])
    assert m.limit(Matrix([[1], [0]])) == approx(a / c)
    assert m.limit(Matrix([[0], [1]])) == approx(b / d)
    assert m.limit(Matrix([[1], [1]])) == approx((a + b) / (c + d))


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


def test_as_pcf():
    from ramanujan import PCF
    from ramanujan.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1.subs([[c0, 0], [c1, 1], [c2, 1], [c3, 3]])
    matrix = cmf.trajectory_matrix([1, 1], [1, 1])
    pcf = matrix.as_pcf()
    assert pcf.simplify() == PCF(5 + 10 * n, 1 - 9 * n**2)


def test_as_pcf_parametric():
    from sympy.abc import c
    from ramanujan import PCF
    from ramanujan.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1.subs([[c0, 0], [c1, 1], [c2, 1], [c3, c]])
    matrix = cmf.trajectory_matrix({x: 1, y: 1}, {x: 1, y: 1})
    pcf = matrix.as_pcf(True)
    print(pcf)
    assert pcf.simplify() == PCF((1 + 2 * n) * (c + 2), 1 - (c * n) ** 2)
