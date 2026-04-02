from pytest import approx

import sympy as sp
from sympy.abc import n, z

from ramanujantools import Position, Matrix, LinearRecurrence
from ramanujantools.cmf import MeijerG


def test_conserving():
    for _p in range(1, 3):
        for _q in range(1, 3):
            for _n in range(_p):
                for _m in range(_q):
                    MeijerG(_m, _n, _p, _q, z).validate_conserving()


def test_gamma():
    cmf = MeijerG(3, 2, 2, 3, 1)
    a0, a1 = sp.symbols("a:2")
    b0, b1, b2 = sp.symbols("b:3")
    start = Position({a0: 0, a1: 0, b0: 0, b1: 0, b2: 0})
    trajectory = Position({a0: -1, a1: -1, b0: 1, b1: 1, b2: 1})
    m = cmf.trajectory_matrix(trajectory, start)

    mm = m.inverse().transpose().as_polynomial()
    limit = mm.limit({n: 1}, 200, {n: 0})
    limit.initial_values = Matrix([[1, 1, 0], [1, 1, 1]])

    assert limit.as_float() == approx(limit.mp.euler)


def test_asymptotics_fail1():
    cmf = MeijerG(3, 2, 2, 3, 1)
    a0, a1 = sp.symbols("a:2")
    b0, b1, b2 = sp.symbols("b:3")
    start = Position({a0: 0, a1: 0, b0: 0, b1: 0, b2: 0})
    trajectory = Position({a0: 0, a1: 0, b0: 0, b1: 1, b2: 1})
    m = cmf.trajectory_matrix(trajectory, start)
    r = LinearRecurrence(m)

    expected = [
        n**4 * sp.factorial(n) ** 2,
        (-1) ** n
        * n ** (sp.Rational(11, 4))
        * sp.exp(2 * sp.I * sp.sqrt(n))
        * sp.factorial(n),
        (-1) ** n
        * n ** (sp.Rational(11, 4))
        * sp.exp(-2 * sp.I * sp.sqrt(n))
        * sp.factorial(n),
    ]
    assert expected == r.asymptotics(precision=10)


def test_asymptotics_fail2():
    cmf = MeijerG(3, 2, 2, 3, 1)
    a0, a1 = sp.symbols("a:2")
    b0, b1, b2 = sp.symbols("b:3")
    start = Position({a0: 0, a1: 0, b0: 0, b1: 0, b2: 0})
    trajectory = Position({a0: 0, a1: 0, b0: 0, b1: 0, b2: 1})
    m = cmf.trajectory_matrix(trajectory, start)
    r = LinearRecurrence(m)

    expected = [n**2 * sp.log(n) * sp.factorial(n), n**2 * sp.factorial(n), 1]
    assert expected == r.asymptotics(precision=11)


def test_asymptotics_fail3():
    cmf = MeijerG(3, 2, 2, 3, 1)
    a0, a1 = sp.symbols("a:2")
    b0, b1, b2 = sp.symbols("b:3")
    start = Position({a0: 0, a1: 0, b0: 0, b1: 0, b2: 0})
    trajectory = Position({a0: 0, a1: 0, b0: 1, b1: 2, b2: 2})
    m = cmf.trajectory_matrix(trajectory, start)
    r = LinearRecurrence(m)

    expected = [
        n**10 * sp.factorial(n) ** 4,
        (-16) ** n
        * n ** sp.Rational(31, 4)
        * sp.exp(4 * sp.I * sp.sqrt(n))
        * sp.factorial(n) ** 3,
        (-16) ** n
        * n ** sp.Rational(31, 4)
        * sp.exp(-4 * sp.I * sp.sqrt(n))
        * sp.factorial(n) ** 3,
    ]

    assert expected == r.asymptotics(precision=18)


def test_asymptotics_euler_trajectory():
    cmf = MeijerG(3, 2, 2, 3, 1)
    a0, a1 = sp.symbols("a:2")
    b0, b1, b2 = sp.symbols("b:3")
    start = Position({a0: 0, a1: 0, b0: 0, b1: 0, b2: 0})
    trajectory = Position({a0: 0, a1: 0, b0: 1, b1: 1, b2: 1})
    m = cmf.trajectory_matrix(trajectory, start)
    r = LinearRecurrence(m)

    expected = [
        n ** (sp.Rational(16, 3))
        * sp.exp(
            -sp.I
            * n ** (sp.Rational(1, 3))
            * (6 * n ** (sp.Rational(1, 3)) + 1 - sp.sqrt(3) * sp.I)
            / (sp.sqrt(3) - sp.I)
        )
        * sp.factorial(n) ** 2,
        n ** (sp.Rational(16, 3))
        * sp.exp(
            n ** (sp.Rational(1, 3))
            * (
                3 * sp.sqrt(3) * n ** (sp.Rational(1, 3))
                + 3 * sp.I * n ** (sp.Rational(1, 3))
                + 2 * sp.I
            )
            / (sp.sqrt(3) - sp.I)
        )
        * sp.factorial(n) ** 2,
        n ** (sp.Rational(16, 3))
        * sp.exp(-(n ** (sp.Rational(1, 3))) * (3 * n ** (sp.Rational(1, 3)) - 1))
        * sp.factorial(n) ** 2,
    ]
    actual = r.asymptotics(precision=12)
    assert expected == actual
