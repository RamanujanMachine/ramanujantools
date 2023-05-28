from pytest import raises
from sympy.abc import x, y

from ramanujan import Matrix, simplify, CMF, known_cmfs


DEFAULT_START = {x: x, y: y}


def test_non_conserving_throws():
    m = Matrix([[x, x + 17], [y * x, y * 3 - x + 5]])
    with raises(ValueError):
        CMF(m, m)


def test_trajectory_matrix_axis():
    cmf = known_cmfs.e
    assert cmf.trajectory_matrix([3, 0]) == simplify(
        cmf.Mx(x, y).walk({x: 1, y: 0}, 3, DEFAULT_START)
    )
    assert cmf.trajectory_matrix([0, 2]) == simplify(
        cmf.My(x, y).walk({x: 0, y: 1}, 2, DEFAULT_START)
    )


def test_trajectory_matrix_diagonal():
    cmf = known_cmfs.e
    assert cmf.trajectory_matrix([1, 1]) == simplify(cmf.Mx(x, y) * cmf.My(x + 1, y))


def test_walk_axis():
    cmf = known_cmfs.e
    assert cmf.walk([1, 0], 17) == cmf.Mx(x, y).walk({x: 1, y: 0}, 17, {x: 1, y: 1})
    assert cmf.walk([0, 1], 17) == cmf.My(x, y).walk({x: 0, y: 1}, 17, {x: 1, y: 1})


def test_walk_diagonal():
    cmf = known_cmfs.e
    Mxy = cmf.trajectory_matrix([1, 1])
    assert cmf.walk([1, 1], 17) == Mxy.walk({x: 1, y: 1}, 17 // 2, {x: 1, y: 1})
