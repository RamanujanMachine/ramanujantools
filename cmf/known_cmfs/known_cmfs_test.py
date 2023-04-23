from pytest import approx
from math import e, pi, log
from scipy.special import zeta

from cmf import Matrix, CMF, known_cmfs
from cmf.known_cmfs import c0, c1, c2, c3


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
    trajectory = [1, 1]
    for a, b in zip(range(1, 10), range(1, 10)):
        print(trajectory, [a, b])
        assert cmf.subs([[c0, 0], [c1, a], [c2, 0], [c3, b]]).limit(
            trajectory, 100
        ) == approx(-a + b / log(1 + a / b), 10e-4)
