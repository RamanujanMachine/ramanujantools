from pytest import raises
from sympy.abc import x, y, n

from ramanujan import Matrix, simplify
from ramanujan.cmf import CMF, known_cmfs


DEFAULT_START = {x: x, y: y}


def test_non_conserving_throws():
    m = Matrix([[x, x + 17], [y * x, y * 3 - x + 5]])
    with raises(ValueError):
        CMF(m, m)


def test_trajectory_matrix_axis():
    cmf = known_cmfs.e
    assert cmf.trajectory_matrix([3, 0]) == simplify(
        cmf.Mx.walk({x: 1, y: 0}, 3, DEFAULT_START)
    )
    assert cmf.trajectory_matrix([0, 2]) == simplify(
        cmf.My.walk({x: 0, y: 1}, 2, DEFAULT_START)
    )


def test_trajectory_matrix_diagonal():
    cmf = known_cmfs.e
    assert cmf.trajectory_matrix([1, 1]) == simplify(cmf.Mx * cmf.My({x: x + 1}))


def test_trajectory_matrix_diagonal_substitute():
    from sympy.abc import n

    cmf = known_cmfs.e
    assert cmf.trajectory_matrix([1, 1], [3, 5]) == simplify(
        cmf.trajectory_matrix([1, 1]).subs([(x, n + 2), (y, n + 4)])
    )


def test_walk_axis():
    cmf = known_cmfs.e
    assert cmf.walk([1, 0], 17) == cmf.Mx.walk({x: 1, y: 0}, 17, {x: 1, y: 1})
    assert cmf.walk([0, 1], 17) == cmf.My.walk({x: 0, y: 1}, 17, {x: 1, y: 1})


def test_walk_diagonal():
    cmf = known_cmfs.e
    Mxy = cmf.trajectory_matrix([1, 1])
    assert cmf.walk([1, 1], 17) == Mxy.walk({x: 1, y: 1}, 17 // 2, {x: 1, y: 1})


def test_substitute_trajectory_axis():
    cmf = known_cmfs.e
    assert CMF.substitute_trajectory(cmf.Mx, [1, 0], [1, 0]) == cmf.Mx({x: n, y: 0})
    assert CMF.substitute_trajectory(cmf.My, [0, 1], [0, 1]) == cmf.My({x: 0, y: n})


def test_substitute_trajectory_diagonal():
    m = known_cmfs.e.trajectory_matrix([1, 2])
    assert CMF.substitute_trajectory(m, [1, 2], [3, 5]) == m.subs(
        [(x, n + 2), (y, 2 * n + 3)]
    )


def test_substitute_trajectory_walk_equivalence():
    cmf = known_cmfs.e
    iterations = 7
    trajectory = [1, 1]
    start = [3, 5]
    unsubbed = cmf.trajectory_matrix(trajectory)
    subbed = cmf.trajectory_matrix(trajectory, start)
    assert subbed.walk({n: 1}, iterations, {n: 1}) == unsubbed.walk(
        {x: trajectory[0], y: trajectory[1]}, iterations, {x: start[0], y: start[1]}
    )
