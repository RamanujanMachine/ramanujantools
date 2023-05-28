from pytest import approx

import itertools

from math import e, pi, log
from scipy.special import zeta

from ramanujan import Matrix, CMF, known_cmfs
from ramanujan.known_cmfs import c0, c1, c2, c3


def test_cmf_e():
    cmf = known_cmfs.e
    assert cmf.limit([1, 1], 100, [0, 0]) == approx(1 / (1 - e))


def test_cmf_pi():
    cmf = known_cmfs.pi
    assert cmf.limit([1, 1], 100, [1, 0]) == approx((2 - pi) / 2)


def test_cmf_zeta3():
    cmf = known_cmfs.zeta3
    assert cmf.limit([1, 1], 100, [1, 1]) == approx((1 - zeta(3)) / zeta(3))


def test_cmf1():
    cmf = known_cmfs.cmf1
    for a, b in itertools.product(range(1, 10), range(1, 10)):
        assert cmf.subs([[c0, 0], [c1, a], [c2, 0], [c3, b]]).limit(
            [1, 1], 100
        ) == approx(-a + b / log(1 + b / a), 1e-4)
