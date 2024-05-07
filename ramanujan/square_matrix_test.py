import pytest

import sympy as sp
from sympy.abc import x, y

from ramanujan import SquareMatrix, simplify, zero, inf


def test_asserts_squared():
    with pytest.raises(AssertionError):
        SquareMatrix([1, 2, 3, 4])
    SquareMatrix([[1, 2], [3, 4]])


def test_gcd():
    a = 2 * 3 * 5
    b = 2 * 3 * 7
    c = 2 * 5 * 7
    d = 3 * 5 * 7
    m = SquareMatrix([[a, b], [c, d]])
    m *= 11
    assert 11 == m.gcd()


def test_normalize():
    initial = SquareMatrix([[2, 3], [5, 7]])
    gcd = sp.Rational(17, 13)
    m = gcd * initial
    assert m.gcd() == gcd
    assert m.normalize() == initial


def test_inverse():
    a = 5
    b = 2
    c = 3
    d = 7
    m = SquareMatrix([[a, b], [c, d]])
    expected = SquareMatrix([[d, -b], [-c, a]]) / (a * d - b * c)
    assert expected == m.inverse()


def test_walk_0():
    m = SquareMatrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 0, y: 1}, 0, {x: x, y: y}) == SquareMatrix.eye(2)


def test_walk_1():
    m = SquareMatrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk({x: 1, y: 0}, 1, {x: x, y: y}) == m


def test_walk_list():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29, 53, 99]
    m = SquareMatrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk(trajectory, iterations, start) == [
        m.walk(trajectory, i, start) for i in iterations
    ]


def test_walk_sequence():
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29, 53, 99]
    m = SquareMatrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert m.walk(trajectory, tuple(iterations), start) == m.walk(
        trajectory, set(iterations), start
    )


def test_walk_axis():
    m = SquareMatrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
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
    m = SquareMatrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 1, y: 1}, 4, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 2, y: 2}) * m({x: 3, y: 3}) * m({x: 4, y: 4})
    )
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 1, y: 1})) == simplify(
        m({x: 1, y: 1}) * m({x: 4, y: 3}) * m({x: 7, y: 5})
    )


def test_walk_different_start():
    m = SquareMatrix([[x, 3 * x + 5 * y], [y**7 + x - 3, x**5]])
    assert simplify(m.walk({x: 3, y: 2}, 3, {x: 5, y: 7})) == simplify(
        m({x: 5, y: 7}) * m({x: 8, y: 9}) * m({x: 11, y: 11})
    )


def test_zero_vector():
    assert zero() == sp.Matrix([0, 1])


def test_inf_vector():
    assert inf() == sp.Matrix([1, 0])
