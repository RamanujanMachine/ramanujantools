from pytest import approx
from math import e, pi
from scipy.special import zeta

from matrix import Matrix
from cmf import CMF
import known_cmfs


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


def test_cmf2():
    cmf = known_cmfs.cmf1
