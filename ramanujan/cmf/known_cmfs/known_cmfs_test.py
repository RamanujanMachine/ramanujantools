from pytest import approx

import itertools
from math import e, pi, log

from mpmath import mp, zeta

from sympy.abc import x, y, n

from ramanujan.cmf import known_cmfs
from ramanujan.pcf import PCF


def test_cmf_e():
    cmf = known_cmfs.e()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 0, y: 0}).as_float() == approx(1 / (1 - e))


def test_cmf_pi():
    cmf = known_cmfs.pi()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 1, y: 0}).as_float() == approx((2 - pi) / 2)


def test_cmf_zeta3():
    cmf = known_cmfs.zeta3()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 1, y: 1}).as_float() == approx(
        (1 - zeta(3)) / zeta(3)
    )


def test_apery():
    cmf = known_cmfs.zeta3()
    pcf = cmf.as_pcf({x: 1, y: 1}).pcf
    # This is Apery's PCF
    assert pcf == PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))

    depth = 2000
    actual_limit = pcf.limit(depth)
    actual_limit.increase_precision()  # Must before expected_limit, to ensure precision for zeta(3)
    expected_limit = mp.mpf(6 / zeta(3))
    assert actual_limit.as_float() == approx(float(expected_limit))
    delta = pcf.delta(depth, expected_limit)
    assert delta > 0.086 and delta < 0.87


def test_cmf1():
    from ramanujan.cmf.known_cmfs import c0, c1, c2, c3

    cmf = known_cmfs.cmf1()
    for a, b in itertools.product(range(1, 10), range(1, 10)):
        assert cmf.subs([[c0, 0], [c1, a], [c2, 0], [c3, b]]).limit(
            {x: 1, y: 1}, 100
        ).as_float() == approx(-a + b / log(1 + b / a), 1e-4)


def test_all_cmfs():
    r"""
    Checks that all cmfs are indeed cmfs.
    If one of these is not a cmf, an exception will be thrown.
    """
    known_cmfs.e()
    known_cmfs.pi()
    known_cmfs.zeta3()
    known_cmfs.var_root_cmf()
    known_cmfs.cmf1()
    known_cmfs.cmf2()
    known_cmfs.cmf3_1()
    known_cmfs.cmf3_2()
    known_cmfs.cmf3_3()
    known_cmfs.hypergeometric_derived_3d()


def test_back_conserving():
    cmf = known_cmfs.hypergeometric_derived_3d()
    cmf.assert_conserving(check_negatives=True)
