from sympy.abc import x, y

from ramanujan.cmf.ffbar import FFbar
from ramanujan.cmf import known_cmfs


def test_linear_condition():
    assert FFbar.linear_condition(x + y, x - y) == 0
    assert FFbar.linear_condition(2**x - 3**y, 2**x + 3**y) != 0


def test_quadratic_condition():
    assert FFbar.quadratic_condition(x**2 + x * y + y**2, x - y) == 0
    assert FFbar.quadratic_condition(x**2 + x * y + y**2, x - y + 1) != 0


def test_load_cmf1():
    known_cmfs.cmf1


def test_load_cmf2():
    known_cmfs.cmf2


def test_load_cmf3():
    known_cmfs.cmf3_1
    known_cmfs.cmf3_2
    known_cmfs.cmf3_3
