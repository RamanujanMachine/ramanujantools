from pytest import raises
from sympy.abc import a, b, c, x, y, n

from ramanujan import Limit, SquareMatrix, simplify
from ramanujan.cmf import CMF, known_cmfs


def test_non_conserving_throws():
    m = SquareMatrix([[x, x + 17], [y * x, y * 3 - x + 5]])
    with raises(ValueError):
        CMF(matrices={x: m, y: m})


def test_symbols():
    cmf = known_cmfs.cmf1()
    expected_axes = {x, y}
    expected_parameters = {known_cmfs.c0, known_cmfs.c1, known_cmfs.c2, known_cmfs.c3}
    assert expected_axes == cmf.axes()
    assert expected_parameters == cmf.parameters()
    assert set().union(expected_axes, expected_parameters) == cmf.free_symbols()


def test_trajectory_matrix_axis():
    cmf = known_cmfs.e()
    assert cmf.trajectory_matrix({x: 3, y: 0}) == simplify(
        cmf.M(x).walk({x: 1, y: 0}, 3, {x: x, y: y})
    )
    assert cmf.trajectory_matrix({x: 0, y: 2}) == simplify(
        cmf.M(y).walk({x: 0, y: 1}, 2, {x: x, y: y})
    )


def test_trajectory_matrix_diagonal():
    cmf = known_cmfs.e()
    assert cmf.trajectory_matrix({x: 1, y: 1}) == simplify(
        cmf.M(x) * cmf.M(y)({x: x + 1})
    )


def test_back_negates_forward():
    cmf = known_cmfs.e()
    assert (
        SquareMatrix.eye(2)
        == (cmf.M(x, True) * cmf.M(x, False).subs({x: x + 1})).normalize()
    )
    assert (
        SquareMatrix.eye(2)
        == (cmf.M(x, False) * cmf.M(x, True).subs({x: x - 1})).normalize()
    )
    assert (
        SquareMatrix.eye(2)
        == (cmf.M(y, True) * cmf.M(y, False).subs({y: y + 1})).normalize()
    )
    assert (
        SquareMatrix.eye(2)
        == (cmf.M(y, False) * cmf.M(y, True).subs({y: y - 1})).normalize()
    )


def test_trajectory_matrix_negative_axis():
    cmf = known_cmfs.e()
    assert (
        cmf.trajectory_matrix({x: -3, y: 0})
        == cmf.M(x, False).walk({x: -1, y: 0}, 3, {x: x, y: y}).normalize()
    )
    assert (
        cmf.trajectory_matrix({x: 0, y: -2})
        == cmf.M(y, False).walk({x: 0, y: -1}, 2, {x: x, y: y}).normalize()
    )


def test_trajectory_matrix_negative():
    cmf = known_cmfs.hypergeometric_dervied_2F1()
    expected = (
        cmf.M(a, sign=True)
        * cmf.M(b, sign=False).subs({a: a + 1})
        * cmf.M(b, sign=False).subs({a: a + 1, b: b - 1})
        * cmf.M(c, sign=False).subs({a: a + 1, b: b - 2})
    )

    assert expected.normalize() == cmf.trajectory_matrix({a: 1, b: -2, c: -1})


def test_trajectory_matrix_diagonal_substitute():
    from sympy.abc import n

    cmf = known_cmfs.e()
    assert cmf.trajectory_matrix({x: 1, y: 1}, {x: 3, y: 5}) == simplify(
        cmf.trajectory_matrix({x: 1, y: 1}).subs([(x, n + 2), (y, n + 4)])
    )


def test_walk_axis():
    cmf = known_cmfs.e()
    assert cmf.walk({x: 1, y: 0}, 17) == cmf.M(x).walk({x: 1, y: 0}, 17, {x: 1, y: 1})
    assert cmf.walk({x: 0, y: 1}, 17) == cmf.M(y).walk({x: 0, y: 1}, 17, {x: 1, y: 1})


def test_walk_diagonal():
    cmf = known_cmfs.e()
    Mxy = cmf.trajectory_matrix({x: 1, y: 1}, {x: 1, y: 1})
    assert cmf.walk({x: 1, y: 1}, 17) == Mxy.walk({n: 1}, 17 // 2, {n: 1})


def test_limit_diagonal():
    cmf = known_cmfs.e()
    Mxy = cmf.trajectory_matrix({x: 1, y: 1})
    assert cmf.limit({x: 1, y: 1}, 17) == Limit(
        Mxy.walk({x: 1, y: 1}, 17 // 2, {x: 1, y: 1})
    )


def test_walk_list():
    cmf = known_cmfs.e()
    trajectory = {x: 2, y: 3}
    iterations = list(map(lambda x: x * sum(trajectory.values()), [1, 2, 3, 17, 29]))
    assert cmf.walk(trajectory, iterations) == [
        cmf.walk(trajectory, i) for i in iterations
    ]


def test_substitute_trajectory_axis():
    cmf = known_cmfs.e()
    assert CMF.substitute_trajectory(cmf.M(x), {x: 1, y: 0}, {x: 1, y: 0}) == cmf.M(x)(
        {x: n, y: 0}
    )
    assert CMF.substitute_trajectory(cmf.M(y), {x: 0, y: 1}, {x: 0, y: 1}) == cmf.M(y)(
        {x: 0, y: n}
    )


def test_substitute_trajectory_diagonal():
    m = known_cmfs.e().trajectory_matrix({x: 1, y: 2})
    assert CMF.substitute_trajectory(m, {x: 1, y: 2}, {x: 3, y: 5}) == m.subs(
        [(x, n + 2), (y, 2 * n + 3)]
    )


def test_N():
    cmf = known_cmfs.hypergeometric_dervied_2F1()
    assert 2 == cmf.N()


def test_dim():
    cmf = known_cmfs.hypergeometric_dervied_2F1()
    assert 3 == cmf.dim()


def test_substitute_trajectory_walk_equivalence():
    cmf = known_cmfs.e()
    iterations = 7
    trajectory = {x: 1, y: 1}
    start = {x: 3, y: 5}
    unsubbed = cmf.trajectory_matrix(trajectory)
    subbed = cmf.trajectory_matrix(trajectory, start)
    assert subbed.walk({n: 1}, iterations, {n: 1}) == unsubbed.walk(
        trajectory, iterations, start
    )
