import sympy as sp

from ramanujantools.cmf import CMF
from ramanujantools.cmf.known_cmfs import e, pFq


def test_trajectory_matrix_simple(benchmark):
    from sympy.abc import x, y

    cmf = e()
    start = {x: 1, y: 1}
    trajectory = {x: 1, y: 1}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_2f2_euler(benchmark):
    x0, x1 = sp.symbols("x:2")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(2, 2, -1)
    start = {x0: 1, x1: 1, y0: -1, y1: -1}
    trajectory = {x0: 1, x1: 1, y0: 0, y1: -1}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_2f2_deep(benchmark):
    x0, x1 = sp.symbols("x:2")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(2, 2, -1)
    start = {x0: 1, x1: 1, y0: -1, y1: 1}
    trajectory = {x0: 12, x1: 13, y0: -14, y1: 20}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_3f2(benchmark):
    x0, x1, x2 = sp.symbols("x:3")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(3, 2, 1)
    start = {x0: 2, x1: 2, x2: 2, y0: 4, y1: 4}
    trajectory = {x0: 5, x1: 5, x2: 5, y0: 10, y1: 10}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_3f2_rational(benchmark):
    x0, x1, x2 = sp.symbols("x:3")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(3, 2, 1)
    start = {
        x0: sp.Rational(1, 2),
        x1: sp.Rational(1, 2),
        x2: sp.Rational(1, 2),
        y0: sp.Rational(3, 2),
        y1: sp.Rational(3, 2),
    }
    trajectory = {x0: 5, x1: 5, x2: 5, y0: 10, y1: 10}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_3f2_pertubated(benchmark):
    x0, x1, x2 = sp.symbols("x:3")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(3, 2, 1)
    start = {x0: 2, x1: 2, x2: 2, y0: 4, y1: 4}
    trajectory = {x0: 5, x1: 6, x2: 5, y0: 10, y1: 11}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_4f3(benchmark):
    x0, x1, x2, x3 = sp.symbols("x:4")
    y0, y1, y2 = sp.symbols("y:3")
    cmf = pFq(4, 3, 1)
    start = {x0: 1, x1: 1, x2: 2, x3: 2, y0: 3, y1: 3, y2: 4}
    trajectory = {x0: 1, x1: 1, x2: 2, x3: 2, y0: 3, y1: 3, y2: 4}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_4f3_huge(benchmark):
    x0, x1, x2, x3 = sp.symbols("x:4")
    y0, y1, y2 = sp.symbols("y:3")
    cmf = pFq(4, 3, 1)
    trajectory = {x0: -1, x1: 1, x2: 2, x3: -2, y0: 3, y1: 7, y2: 4}
    start = trajectory
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_walk_4f3(benchmark):
    x0, x1, x2, x3 = sp.symbols("x:4")
    y0, y1, y2 = sp.symbols("y:3")
    cmf = pFq(4, 3, 1)
    start = {x0: 1, x1: 1, x2: 2, x3: 2, y0: 3, y1: 3, y2: 4}
    trajectory = {x0: 1, x1: 1, x2: 2, x3: 2, y0: 3, y1: 3, y2: 4}
    benchmark(CMF.walk, cmf, trajectory, 1000, start)
