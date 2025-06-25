from pytest import approx

import itertools

import sympy as sp
from sympy.abc import x, y, n

from ramanujantools.pcf import PCF
from ramanujantools.cmf import known_cmfs


def test_cmf_e():
    cmf = known_cmfs.e()
    limit = cmf.limit({x: 1, y: 1}, 50, {x: 0, y: 0})
    assert limit.as_float() == approx(1 / (1 - limit.mp.e))


def test_cmf_pi():
    cmf = known_cmfs.pi()
    limit = cmf.limit({x: 1, y: 1}, 50, {x: 1, y: 0})
    assert limit.as_float() == approx((2 - limit.mp.pi) / 2)


def test_cmf_zeta3():
    cmf = known_cmfs.zeta3()
    limit = cmf.limit({x: 1, y: 1}, 50, {x: 1, y: 1})
    assert limit.as_float() == approx((1 - limit.mp.zeta(3)) / limit.mp.zeta(3))


def test_apery():
    cmf = known_cmfs.zeta3()
    pcf = PCF(cmf.trajectory_matrix({x: 1, y: 1}, {x: 0, y: 0})).deflate_all()
    # This is Apery's PCF
    assert pcf == PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))

    depth = 2000
    actual_limit = pcf.limit(depth)
    ctx = actual_limit.mp
    expected_limit = ctx.mpf(6 / ctx.zeta(3))
    assert actual_limit.as_float() == approx(float(expected_limit))
    delta = pcf.delta(depth, expected_limit)
    assert delta > 0.08


def test_cmf1():
    c0, c1, c2, c3 = sp.symbols("c:4")

    cmf = known_cmfs.cmf1()
    for a, b in itertools.product(range(1, 10), range(1, 10)):
        limit = cmf.subs({c0: 0, c1: a, c2: 0, c3: b}).limit(
            {x: 1, y: 1}, 50, {x: 1, y: 1}
        )
        assert limit.as_float() == approx(-a + b / limit.mp.log(1 + b / a), 1e-4)


def test_all_conserving():
    r"""
    Checks that all cmfs are indeed cmfs.
    If one of these is not a cmf, an exception will be thrown.
    """
    known_cmfs.e().assert_conserving()
    known_cmfs.pi().assert_conserving()
    known_cmfs.symmetric_pi().assert_conserving()
    known_cmfs.zeta3().assert_conserving()
    known_cmfs.var_root_cmf().assert_conserving()
    known_cmfs.cmf1().assert_conserving()
    known_cmfs.cmf2().assert_conserving()
    known_cmfs.cmf3_1().assert_conserving()
    known_cmfs.cmf3_2().assert_conserving()
    known_cmfs.cmf3_3().assert_conserving()
    known_cmfs.hypergeometric_derived_2F1().assert_conserving()
    known_cmfs.hypergeometric_derived_3F2().assert_conserving()


def test_back_conserving():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    cmf.assert_conserving(check_negatives=True)
