from pytest import approx
from matrix import Position, Matrix
from sympy.abc import x, y


def test_position_add():
    a = [5, 7]
    b = [17, 63]
    x = Position(a)
    y = Position(b)
    x_y = Position([a[0] + b[0], a[1] + b[1]])
    assert x + y == x_y


def test_position_iadd():
    a = [57, 29]
    b = [82, 31]
    x = Position(a)
    y = Position(b)
    x_y = Position([a[0] + b[0], a[1] + b[1]])
    x += y
    assert x == x_y


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
    assert m.walk([0, 1], 0) == Matrix.eye(2)


def test_walk_1():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk([1, 0], 1) == m(1, 1)


def test_walk_axis():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk([1, 0], 3).simplify() == (m(1, 1) * m(2, 1) * m(3, 1)).simplify()
    assert (
        m.walk([0, 1], 5).simplify()
        == (m(1, 1) * m(1, 2) * m(1, 3) * m(1, 4) * m(1, 5)).simplify()
    )


def test_walk_diagonal():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert (
        m.walk([1, 1], 4).simplify()
        == (m(1, 1) * m(2, 2) * m(3, 3) * m(4, 4)).simplify()
    )
    assert m.walk([3, 2], 3).simplify() == (m(1, 1) * m(4, 3) * m(7, 5)).simplify()


def test_walk_different_start():
    m = Matrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert (
        m.walk([3, 2], 3, [5, 7]).simplify()
        == (m(5, 7) * m(8, 9) * m(11, 11)).simplify()
    )
