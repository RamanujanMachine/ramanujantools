from sympy.abc import x, y

from ramanujan import ffbar, known_cmfs


def test_linear_condition():
    assert ffbar.linear_condition(x + y, x - y)
    assert not ffbar.linear_condition(2**x - 3**y, 2**x + 3**y)


def test_quadratic_condition():
    assert ffbar.quadratic_condition(x**2 + x * y + y**2, x - y)
    assert not ffbar.quadratic_condition(x**2 + x * y + y**2, x - y + 1)


def test_load_cmf1():
    known_cmfs.cmf1


def test_load_cmf2():
    known_cmfs.cmf2


def test_load_cmf3():
    known_cmfs.cmf3_1
    known_cmfs.cmf3_2
    known_cmfs.cmf3_3
