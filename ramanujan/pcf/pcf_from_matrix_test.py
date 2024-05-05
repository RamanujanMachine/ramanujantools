from sympy.abc import c, n, x, y

from ramanujan import SquareMatrix
from ramanujan.pcf import PCF, PCFFromSquareMatrix


def test_pcf_from_matrix_parametric():
    from ramanujan.cmf.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1().subs([[c0, 0], [c1, 1], [c2, 1], [c3, c]])
    matrix = cmf.trajectory_matrix({x: 1, y: 1}, {x: 1, y: 1})
    assert PCFFromSquareMatrix(matrix).pcf == PCF((1 + 2 * n) * (c + 2), 1 - (c * n) ** 2)


def test_pcf_from_matrix_relative_limit():
    from ramanujan.cmf.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1().subs([[c0, 0], [c1, 1], [c2, 1], [c3, c]])
    matrix = cmf.trajectory_matrix({x: 1, y: 1}, {x: 1, y: 1})
    pcf = PCFFromSquareMatrix(matrix)
    assert pcf.relative_limit() == SquareMatrix([[2, 1], [0, 1]])


def test_as_pcf_equivalent():
    from ramanujan.cmf.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1().subs([[c0, 0], [c1, 1], [c2, 1], [c3, c]])
    matrix = cmf.trajectory_matrix({x: 1, y: 1}, {x: 1, y: 1})
    assert PCFFromSquareMatrix(matrix) == matrix.as_pcf()
