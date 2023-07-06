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


def test_call_1():
    m = Matrix([[x + 1, 2 * x], [x**2, 7 * x - 13]])
    assert m(0, y) == Matrix([[1, 0], [0, -13]])
    assert m(5, y) == Matrix([[6, 10], [25, 22]])
    assert m(x, y) == m(x, y + 1)


def test_call_2():
    m = Matrix([[y + 1, 2 * x * y], [x**2, 7 * x + y]])
    assert m(0, 0) == Matrix([[1, 0], [0, 0]])
    assert m(5, 7) == Matrix([[8, 70], [25, 42]])
    assert m(x, 1) == Matrix([[2, 2 * x], [x**2, 7 * x + 1]])
    assert m(1, y) == Matrix([[y + 1, 2 * y], [1, 7 + y]])


def test_walk_0():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 0, y: 1}, 0, {x: x, y: y}) == Matrix.eye(2)


def test_walk_1():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 1, y: 0}, 1, {x: x, y: y}) == m(x, y)


def test_walk_axis():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 1, y: 0}, 3, {x: 1, y: 1})) == simplify(
        m(1, 1) * m(2, 1) * m(3, 1)
    )
    assert simplify(m.walk({x: 0, y: 1}, 5, {x: 1, y: 1})) == simplify(
        m(1, 1) * m(1, 2) * m(1, 3) * m(1, 4) * m(1, 5)
    )


def test_walk_diagonal():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 1, y: 1}, 4, {x: 1, y: 1})) == simplify(
        m(1, 1) * m(2, 2) * m(3, 3) * m(4, 4)
    )
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 1, y: 1})) == simplify(
        m(1, 1) * m(4, 3) * m(7, 5)
    )


def test_walk_different_start():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 5, y: 7})) == simplify(
        m(5, 7) * m(8, 9) * m(11, 11)
    )


def test_as_pcf():
    from ramanujan import PCF
    from ramanujan.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1.subs([[c0, 0], [c1, 1], [c2, 1], [c3, 3]])
    matrix = cmf.trajectory_matrix([1, 1], [1, 1])
    pcf = matrix.as_pcf()
    print(pcf)
    assert pcf.simplify() == PCF(5 + 10 * n, 1 - 9 * n**2)
