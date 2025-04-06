import sympy as sp
from sympy.abc import x, y, z

from ramanujantools import Position


def test_add():
    p1 = Position({x: 17, y: 3})
    p2 = Position({y: 4, z: 3})
    assert {x: 17, y: 7, z: 3} == p1 + p2


def test_iadd():
    p = Position({x: 17, y: 3})
    p += Position({y: 4, z: 3})
    assert {x: 17, y: 7, z: 3} == p


def test_isub():
    p = Position({x: 17, y: 3})
    p -= Position({y: 4, z: 3})
    assert {x: 17, y: -1, z: -3} == p


def test_sub():
    p1 = Position({x: 17, y: 3})
    p2 = Position({y: 4, z: 3})
    assert {x: 17, y: -1, z: -3} == p1 - p2


def test_mul():
    p = Position({x: 1, y: -2, z: 3})
    expected = {key: -7 * value for key, value in p.items()}
    assert expected == -7 * p
    assert expected == p * -7


def test_neg():
    p = Position({x: 1, y: -2, z: 3})
    assert {key: -value for key, value in p.items()} == -p


def test_longest():
    p = Position({x: 1, y: 2, z: -3})
    assert 3 == p.longest()


def test_longest_empty():
    p = Position()
    assert 0 == p.longest()


def test_shortest():
    p = Position({x: 1, y: 2, z: -3})
    assert 1 == p.shortest()


def test_shortest_empty():
    p = Position()
    assert 0 == p.longest()


def test_signs():
    p = Position({x: 1, y: 2, z: -3})
    assert Position({x: 1, y: 1, z: -1}) == p.signs()


def test_signs_empty():
    assert Position() == Position().signs()


def test_is_integer():
    assert Position({x: 1, y: 7}).is_integer()
    assert not Position({x: 1, y: x**2 + 3 * x + 1}).is_integer()
    assert not Position({x: x / 2, y: x**2 + 3 * x + 1}).is_integer()
    assert not Position({x: sp.Rational(1, 2), y: 1}).is_integer()


def test_is_polynomial():
    assert Position({x: 1, y: 7}).is_polynomial()
    assert Position({x: 1, y: x**2 + 3 * x + 1}).is_polynomial()
    assert not Position({x: x / 2, y: x**2 + 3 * x + 1}).is_polynomial()
    assert not Position({x: sp.Rational(1, 2), y: 1}).is_polynomial()


def test_is_denominator_lcm():
    assert 1 == Position({x: 1, y: 7}).denominator_lcm()
    assert 1 == Position({x: 1, y: x**2 + 3 * x + 1}).denominator_lcm()
    assert 2 == Position({x: x / 2, y: x**2 + 3 * x + 1}).denominator_lcm()
    assert 15 == Position({x: sp.Rational(1, 3), y: x / 5}).denominator_lcm()


def test_free_symbols():
    assert {x, z} == Position({x: 1, y: z, z: x}).free_symbols()


def test_from_list():
    z0, z1, z2, z3 = sp.symbols("z:4")
    expected = Position({z0: 9, z1: 8, z2: 7, z3: 6})
    actual = Position.from_list([9, 8, 7, 6], "z")
    assert isinstance(actual, Position)
    assert expected == actual
