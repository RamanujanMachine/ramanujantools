from pytest import approx

import sympy as sp
from sympy.abc import n, z

from ramanujantools import Position, Matrix
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
