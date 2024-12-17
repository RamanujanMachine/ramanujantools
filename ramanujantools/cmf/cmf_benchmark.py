import sympy as sp
from sympy.abc import x, y

from ramanujantools.cmf import CMF
from ramanujantools.cmf.known_cmfs import e, pFq


def test_trajectory_matrix_simple(benchmark):
    cmf = e()
    start = {x: 1, y: 1}
    trajectory = {x: 1, y: 1}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_euler(benchmark):
    x0, x1 = sp.symbols("x:2")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(2, 2, -1)
    start = {x0: 1, x1: 1, y0: -1, y1: -1}
    trajectory = {x0: 1, x1: 1, y0: 0, y1: -1}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_huge(benchmark):
    x0, x1, x2 = sp.symbols("x:3")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(3, 2, 1)
    start = {x0: 2, x1: 2, x2: 2, y0: 4, y1: 4}
    trajectory = {x0: 5, x1: 5, x2: 5, y0: 10, y1: 10}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)


def test_trajectory_matrix_pertubated(benchmark):
    x0, x1, x2 = sp.symbols("x:3")
    y0, y1 = sp.symbols("y:2")
    cmf = pFq(3, 2, 1)
    start = {x0: 2, x1: 2, x2: 2, y0: 4, y1: 4}
    trajectory = {x0: 5, x1: 6, x2: 5, y0: 10, y1: 11}
    benchmark(CMF.trajectory_matrix, cmf, trajectory, start)
