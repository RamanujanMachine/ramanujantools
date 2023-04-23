from pytest import raises
from sympy.abc import x, y

from cmf import Matrix, simplify, CMF, known_cmfs

def test_non_conserving_throws():
    m = Matrix([[x, x + 17], [y * x, y * 3 - x + 5]])
    with raises(ValueError):
        CMF(m, m)


def test_trajectory_matrix_axis():
    cmf = known_cmfs.e
    assert cmf.trajectory_matrix([3, 0]) == simplify(cmf.Mx(x, y).walk([1, 0], 3))
    assert cmf.trajectory_matrix([0, 2]) == simplify(cmf.My(x, y).walk([0, 1], 2))


def test_trajectory_matrix_diagonal():
    cmf = known_cmfs.e
    assert cmf.trajectory_matrix([1, 1]) == simplify(cmf.Mx(x, y) * cmf.My(x + 1, y))


def test_walk_axis():
    cmf = known_cmfs.e
    assert cmf.walk([1, 0], 17) == cmf.Mx(x, y).walk([1, 0], 17, [1, 1])
    assert cmf.walk([0, 1], 17) == cmf.My(x, y).walk([0, 1], 17, [1, 1])


def test_walk_axis():
    cmf = known_cmfs.e
    Mxy = cmf.trajectory_matrix([1, 1])
    assert cmf.walk([1, 1], 17) == Mxy.walk([1, 1], 17 // 2, [1, 1])
