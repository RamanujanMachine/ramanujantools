from pytest import approx

import itertools
from math import e, pi, log

from mpmath import mp, zeta

import sympy as sp
from sympy.abc import x, y, z, n

from ramanujantools import Matrix, IntegerRelation
from ramanujantools.pcf import PCF
from ramanujantools.cmf import known_cmfs, CMF


def test_cmf_e():
    cmf = known_cmfs.e()
    assert cmf.limit({x: 1, y: 1}, 50, {x: 0, y: 0}).as_float() == approx(1 / (1 - e))


def test_cmf_pi():
    cmf = known_cmfs.pi()
    assert cmf.limit({x: 1, y: 1}, 50, {x: 1, y: 0}).as_float() == approx((2 - pi) / 2)


def test_cmf_zeta3():
    cmf = known_cmfs.zeta3()
    assert cmf.limit({x: 1, y: 1}, 50, {x: 1, y: 1}).as_float() == approx(
        (1 - zeta(3)) / zeta(3)
    )


def test_apery():
    cmf = known_cmfs.zeta3()
    pcf = cmf.as_pcf({x: 1, y: 1}).pcf
    # This is Apery's PCF
    assert pcf == PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))

    depth = 2000
    actual_limit = pcf.limit(depth)
    with mp.workdps(actual_limit.precision()):
        expected_limit = mp.mpf(6 / zeta(3))
        assert actual_limit.as_float() == approx(float(expected_limit))
        delta = pcf.delta(depth, expected_limit)
        assert delta > 0.08


def test_cmf1():
    from ramanujantools.cmf.known_cmfs import c0, c1, c2, c3

    cmf = known_cmfs.cmf1()
    for a, b in itertools.product(range(1, 10), range(1, 10)):
        assert cmf.subs({c0: 0, c1: a, c2: 0, c3: b}).limit(
            {x: 1, y: 1}, 50
        ).as_float() == approx(-a + b / log(1 + b / a), 1e-4)


def test_2F1_theta_derivative():
    x0 = sp.Symbol("x0")
    x1 = sp.Symbol("x1")
    y0 = sp.Symbol("y0")
    z = sp.Symbol("z")
    expected = CMF(
        {
            x0: Matrix(
                [
                    [1, -x1 * z / (z - 1)],
                    [1 / x0, 1 - (x0 * z + x1 * z - y0 + 1) / (x0 * (z - 1))],
                ]
            ),
            x1: Matrix(
                [
                    [1, -x0 * z / (z - 1)],
                    [1 / x1, 1 - (x0 * z + x1 * z - y0 + 1) / (x1 * (z - 1))],
                ]
            ),
            y0: Matrix(
                [
                    [
                        y0 * (-x0 - x1 + y0) / (x0 * x1 - x0 * y0 - x1 * y0 + y0**2),
                        x0 * x1 * y0 / (x0 * x1 - x0 * y0 - x1 * y0 + y0**2),
                    ],
                    [
                        y0 * (1 - z) / (z * (x0 * x1 - x0 * y0 - x1 * y0 + y0**2)),
                        y0**2 * (z - 1) / (z * (x0 * x1 - x0 * y0 - x1 * y0 + y0**2)),
                    ],
                ]
            ),
        }
    )
    cmf = known_cmfs.pFq(2, 1)
    cmf.assert_conserving()
    assert cmf == expected


def test_2F1_theta_derivative_negate_denominator():
    x0 = sp.Symbol("x0")
    x1 = sp.Symbol("x1")
    y0 = sp.Symbol("y0")
    z = sp.Symbol("z")
    expected = CMF(
        {
            x0: Matrix(
                [
                    [1, -x1 * z / (z - 1)],
                    [1 / x0, 1 - (x0 * z + x1 * z + y0 + 1) / (x0 * (z - 1))],
                ]
            ),
            x1: Matrix(
                [
                    [1, -x0 * z / (z - 1)],
                    [1 / x1, 1 - (x0 * z + x1 * z + y0 + 1) / (x1 * (z - 1))],
                ]
            ),
            y0: Matrix(
                [
                    [1, x0 * x1 * z / ((y0 + 1) * (z - 1))],
                    [
                        -1 / (y0 + 1),
                        1 + (x0 * z + x1 * z + y0 + 1) / ((y0 + 1) * (z - 1)),
                    ],
                ]
            ),
        }
    )
    cmf = known_cmfs.pFq(2, 1, negate_denominator_params=True)
    cmf.assert_conserving()
    assert cmf == expected


def test_2F1_normal_derivative():
    x0 = sp.Symbol("x0")
    x1 = sp.Symbol("x1")
    y0 = sp.Symbol("y0")
    z = sp.Symbol("z")
    expected = CMF(
        {
            x0: Matrix(
                [
                    [1, -x1 / (z - 1)],
                    [z / x0, 1 + (-x0 * z - x1 * z + y0 - 1) / (x0 * (z - 1))],
                ]
            ),
            x1: Matrix(
                [
                    [1, -x0 / (z - 1)],
                    [z / x1, 1 + (-x0 * z - x1 * z + y0 - 1) / (x1 * (z - 1))],
                ]
            ),
            y0: Matrix(
                [
                    [
                        y0 * (-x0 - x1 + y0) / (x0 * x1 - x0 * y0 - x1 * y0 + y0**2),
                        x0 * x1 * y0 / (z * (x0 * x1 - x0 * y0 - x1 * y0 + y0**2)),
                    ],
                    [
                        y0 * (1 - z) / (x0 * x1 - x0 * y0 - x1 * y0 + y0**2),
                        y0**2 * (z - 1) / (z * (x0 * x1 - x0 * y0 - x1 * y0 + y0**2)),
                    ],
                ]
            ),
        }
    )
    cmf = known_cmfs.pFq(2, 1, theta_derivative=False)
    cmf.assert_conserving()
    assert cmf == expected


def test_2F1_normal_derivative_negate_denominator():
    x0 = sp.Symbol("x0")
    x1 = sp.Symbol("x1")
    y0 = sp.Symbol("y0")
    z = sp.Symbol("z")
    expected = CMF(
        {
            x0: Matrix(
                [
                    [1, -x1 / (z - 1)],
                    [z / x0, 1 + (-x0 * z - x1 * z - y0 - 1) / (x0 * (z - 1))],
                ]
            ),
            x1: Matrix(
                [
                    [1, -x0 / (z - 1)],
                    [z / x1, 1 + (-x0 * z - x1 * z - y0 - 1) / (x1 * (z - 1))],
                ]
            ),
            y0: Matrix(
                [
                    [1, x0 * x1 / ((y0 + 1) * (z - 1))],
                    [
                        -z / (y0 + 1),
                        1 - (-x0 * z - x1 * z - y0 - 1) / ((y0 + 1) * (z - 1)),
                    ],
                ]
            ),
        }
    )
    cmf = known_cmfs.pFq(2, 1, theta_derivative=False, negate_denominator_params=True)
    cmf.assert_conserving()
    assert cmf == expected


def test_2F1_z_evaluation():
    p = 2
    q = 1
    z_value = -7
    assert known_cmfs.pFq(p, q, z_eval=z_value) == known_cmfs.pFq(p, q).subs(
        {z: z_value}
    )


def test_gamma():
    cmf = known_cmfs.pFq(2, 2, negate_denominator_params=True, z_eval=-1)
    x0, x1 = sp.symbols("x:2")
    y0, y1 = sp.symbols("y:2")
    trajectory = {x0: 1, x1: 1, y0: 1, y1: 0}
    start = {x0: 1, x1: 1, y0: 1, y1: 1}
    limit = cmf.limit(trajectory, 100, start)
    assert IntegerRelation([[1, 3, 0], [-3, -5, 0]]) == limit.identify(mp.euler)


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
    known_cmfs.pFq(2, 2).assert_conserving()  # randomly choosing 2F2


def test_back_conserving():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    cmf.assert_conserving(check_negatives=True)
