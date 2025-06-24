import sympy as sp
from sympy.abc import n, z

from ramanujantools.cmf import CMF
from ramanujantools.cmf import pFq
from ramanujantools import Matrix, Position

x0, x1, x2, x3 = sp.symbols("x:4")
y0, y1, y2 = sp.symbols("y:3")


def test_subs():
    cmf = pFq(2, 1)
    assert pFq(2, 1, -1) == cmf.subs({z: -1})


def test_ascend_symbolic():
    cmf = pFq(2, 1)
    start = Position({x0: 1, x1: 2, y0: 3})
    trajectory = Position({x0: 4, x1: 5, y0: 6})

    expected_cmf = pFq(3, 2)
    expected_start = start + Position({x2: x2, y1: y1})
    expected_trajectory = trajectory + Position({x2: 0, y1: 0})

    assert (expected_cmf, expected_trajectory, expected_start) == cmf.ascend(
        trajectory, start
    )


def test_ascend_zeta2():
    cmf = pFq(3, 2, 1)
    trajectory = Position({x0: 1, x1: 1, x2: 1, y0: 2, y1: 2})
    start = 2 * Position({x0: 1, x1: 1, x2: 1, y0: 2, y1: 2})

    limit = cmf.trajectory_matrix(trajectory, start).limit({n: 1}, 200, {n: 1})
    assert limit.identify(limit.mp.zeta(2)) is not None
    delta = limit.delta(limit.mp.zeta(2))

    a_cmf, a_trajectory, a_start = cmf.ascend(trajectory, start)
    a_limit = (
        a_cmf.trajectory_matrix(a_trajectory, a_start)
        .subs({x3: 3, y2: 3})
        .limit({n: 1}, 200, {n: 1})
    )
    assert a_limit.identify(a_limit.mp.zeta(2), maxcoeff=10**6) is not None
    a_delta = a_limit.delta(a_limit.mp.zeta(2))

    assert abs(delta - a_delta) < 0.01


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
    cmf = pFq(2, 1)
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
    cmf = pFq(2, 1, negate_denominator_params=False)
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
    cmf = pFq(2, 1, theta_derivative=False)
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
    cmf = pFq(2, 1, theta_derivative=False, negate_denominator_params=False)
    cmf.assert_conserving()
    assert cmf == expected


def test_2F1_z_evaluation():
    p = 2
    q = 1
    z_value = -7
    assert pFq(p, q, z_eval=z_value) == pFq(p, q).subs({z: z_value})


def test_gamma():
    cmf = pFq(2, 2, negate_denominator_params=False, z_eval=-1)
    x0, x1 = sp.symbols("x:2")
    y0, y1 = sp.symbols("y:2")
    trajectory = {x0: 1, x1: 1, y0: 1, y1: 0}
    start = {x0: 1, x1: 1, y0: 1, y1: 1}
    limit = cmf.limit(trajectory, 100, start)
    assert Matrix([[1, 3, 0], [3, 5, 0]]) == limit.identify(limit.mp.euler)


def test_pfq_conserving():
    for p in range(1, 3):
        for q in range(1, 3):
            pFq(p, q).assert_conserving()


def test_predict_N():
    for p in range(1, 5):
        for q in range(1, 5):
            for z_eval in [-1, 1]:
                assert pFq(p, q, z_eval=z_eval).N() == pFq.predict_N(p, q, z_eval)


def test_work_numeric_3f2():
    cmf = pFq(3, 2, -1)
    x0, x1, x2 = sp.symbols("x:3")
    y0, y1 = sp.symbols("y:2")
    start = Position(
        {x0: 1, x1: -sp.Rational(1, 2), x2: 0, y0: sp.Rational(5, 3), y1: -1}
    )
    end = Position(
        {x0: 17, x1: -sp.Rational(13, 2), x2: 0, y0: sp.Rational(20, 3), y1: -4}
    )
    trajectory = end - start
    expected = cmf.walk(trajectory, 1, start)
    actual = cmf.work(start, end)
    assert expected == actual


def test_state_vector():
    p = 3
    q = 2
    z_eval = -1
    a_symbols = sp.symbols(f"a:{p}")
    b_symbols = sp.symbols(f"b:{q}")
    state_vector = pFq.state_vector(a_symbols, b_symbols, z_eval)
    assert len(state_vector) == pFq.predict_N(p, q, z_eval)
    assert state_vector.free_symbols == set(a_symbols).union(b_symbols)
    value = sp.hyper(a_symbols, b_symbols, z)
    for i in range(len(state_vector)):
        assert sp.hyperexpand(value.subs({z: z_eval})) == state_vector[i]
        value = pFq.theta_derivative(value)


def test_evaluate_2f1():
    for z_eval in [
        -1,
        -sp.Rational(1, 2),
        -sp.Rational(1, 4),
        sp.Rational(1, 4),
        sp.Rational(1, 2),
    ]:
        for a_values, b_values in [
            ([1, 1], [2]),
            ([1, 3], [5]),
            ([sp.Rational(3, 2), sp.Rational(5, 2)], [sp.Rational(7, 2)]),
            ([sp.Rational(3, 2), 4], [-sp.Rational(1, 2)]),
        ]:
            evaluation = pFq.evaluate(a_values, b_values, z_eval).simplify()
            expected = sp.hyper(a_values, b_values, z_eval).simplify().simplify()
            assert expected == evaluation
