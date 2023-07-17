from pytest import approx

import itertools
from math import e, pi, log

from scipy.special import zeta
from sympy.abc import x, y

from ramanujan.cmf import known_cmfs


def test_cmf_e():
    cmf = known_cmfs.e()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 0, y: 0}) == approx(1 / (1 - e))


def test_cmf_pi():
    cmf = known_cmfs.pi()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 1, y: 0}) == approx((2 - pi) / 2)


def test_cmf_zeta3():
    cmf = known_cmfs.zeta3()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 1, y: 1}) == approx((1 - zeta(3)) / zeta(3))


def test_cmf1():
    from ramanujan.cmf.known_cmfs import c0, c1, c2, c3

    cmf = known_cmfs.cmf1()
    for a, b in itertools.product(range(1, 10), range(1, 10)):
        assert cmf.subs([[c0, 0], [c1, a], [c2, 0], [c3, b]]).limit(
            {x: 1, y: 1}, 100
        ) == approx(-a + b / log(1 + b / a), 1e-4)
