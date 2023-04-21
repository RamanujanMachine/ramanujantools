from pytest import raises
from sympy.abc import x, y

from matrix import Matrix, simplify
from cmf import CMF


def test_non_conserving_throws():
    m = Matrix([[x, x + 17], [y * x, y * 3 - x + 5]])
    with raises(ValueError):
        CMF(m, m)


def test_trajectory_matrix_axis():
    Mx = Matrix([[1, -y - 1], [-1, x + y + 2]])
    My = Matrix([[0, -y - 1], [-1, x + y + 1]])
    cmf = CMF(Mx, My)
    assert cmf.trajectory_matrix([3, 0]) == simplify(Mx.walk([1, 0], 3))


def test_trajectory_matrix_diagonal():
    Mx = Matrix([[1, -y - 1], [-1, x + y + 2]])
    My = Matrix([[0, -y - 1], [-1, x + y + 1]])
    cmf = CMF(Mx, My)
    assert cmf.trajectory_matrix([1, 1]) == simplify(Mx(x, y) * My(x + 1, y))


def test_walk_axis():
    Mx = Matrix([[1, -y - 1], [-1, x + y + 2]])
    My = Matrix([[0, -y - 1], [-1, x + y + 1]])
    cmf = CMF(Mx, My)
    assert cmf.walk([1, 0], 17) == Mx.walk([1, 0], 17, [1, 1])
    assert cmf.walk([0, 1], 17) == My.walk([0, 1], 17, [1, 1])


def test_walk_axis():
    Mx = Matrix([[1, -y - 1], [-1, x + y + 2]])
    My = Matrix([[0, -y - 1], [-1, x + y + 1]])
    Mxy = Mx(x, y) * My(x + 1, y)
    cmf = CMF(Mx, My)
    assert cmf.walk([1, 1], 17) == Mxy.walk([1, 1], 17 // 2, [1, 1])
