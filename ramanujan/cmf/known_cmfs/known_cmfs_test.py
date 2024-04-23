from pytest import approx

import itertools
from math import e, pi, log

import mpmath as mp
from mpmath import zeta

import sympy as sp
from sympy.abc import x, y, n

from ramanujan.cmf import known_cmfs
from ramanujan.pcf import PCF


def test_cmf_e():
    cmf = known_cmfs.e()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 0, y: 0}).ratio() == approx(1 / (1 - e))


def test_cmf_pi():
    cmf = known_cmfs.pi()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 1, y: 0}).ratio() == approx((2 - pi) / 2)


def test_cmf_zeta3():
    cmf = known_cmfs.zeta3()
    assert cmf.limit({x: 1, y: 1}, 100, {x: 1, y: 1}).ratio() == approx(
        (1 - zeta(3)) / zeta(3)
    )


def test_apery():
    mp.mp.dps = 10000
    cmf = known_cmfs.zeta3()
    pcf = cmf.as_pcf({x: 1, y: 1}).pcf
    # This is Apery's PCF
    assert pcf == PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))

    limit = sp.Float(6 / zeta(3), mp.mp.dps)
    depth = 2000
    assert pcf.limit(depth).ratio() == approx(float(limit))
    delta = pcf.delta(depth, limit)
    assert delta > 0.086 and delta < 0.87


def test_cmf1():
    from ramanujan.cmf.known_cmfs import c0, c1, c2, c3

    cmf = known_cmfs.cmf1()
    for a, b in itertools.product(range(1, 10), range(1, 10)):
        assert cmf.subs([[c0, 0], [c1, a], [c2, 0], [c3, b]]).limit(
            {x: 1, y: 1}, 100
        ).ratio() == approx(-a + b / log(1 + b / a), 1e-4)


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
