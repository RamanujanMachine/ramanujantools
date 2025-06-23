import sympy as sp
from sympy.abc import c, n, x, y

from ramanujantools import Matrix
from ramanujantools.pcf import PCF, PCFFromMatrix
from ramanujantools.cmf.known_cmfs import cmf1

c0, c1, c2, c3 = sp.symbols("c:4")


def test_pcf_from_matrix_parametric():
    cmf = cmf1().subs({c0: 0, c1: 1, c2: 1, c3: c})
    matrix = cmf.trajectory_matrix({x: 1, y: 1}, {x: 0, y: 0})
    assert PCFFromMatrix(matrix).pcf == PCF((1 + 2 * n) * (c + 2), 1 - (c * n) ** 2)


def test_pcf_from_matrix_relative_limit():
    cmf = cmf1().subs({c0: 0, c1: 1, c2: 1, c3: c})
    matrix = cmf.trajectory_matrix({x: 1, y: 1}, {x: 0, y: 0})
    pcf = PCFFromMatrix(matrix)
    assert pcf.relative_limit() == Matrix([[2, 1], [0, 1]])


def test_as_pcf_equivalent():
    cmf = cmf1().subs({c0: 0, c1: 1, c2: 1, c3: c})
    matrix = cmf.trajectory_matrix({x: 1, y: 1}, {x: 0, y: 0})
    assert PCFFromMatrix(matrix) == matrix.as_pcf()
