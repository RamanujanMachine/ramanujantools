from pytest import raises
from sympy.abc import a, b, c, x, y, n

from ramanujantools import Matrix, simplify
from ramanujantools.cmf import CMF, known_cmfs


def test_assert_conserving():
    m = Matrix([[x, x + 17], [y * x, y * 3 - x + 5]])
    cmf = CMF(matrices={x: m, y: m}, validate=False)
    with raises(ValueError):
        cmf.assert_conserving()
    with raises(ValueError):
        cmf = CMF(matrices={x: m, y: m}, validate=True)


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
    assert Matrix.eye(2) == (cmf.M(x, True) * cmf.M(x, False).subs({x: x + 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(x, False) * cmf.M(x, True).subs({x: x - 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(y, True) * cmf.M(y, False).subs({y: y + 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(y, False) * cmf.M(y, True).subs({y: y - 1})).reduce()


def test_trajectory_matrix_negative_axis():
    cmf = known_cmfs.e()
    assert cmf.trajectory_matrix({x: -3, y: 0}).limit_equivalent(
        cmf.M(x, False).walk({x: -1, y: 0}, 3, {x: x, y: y})
    )
    assert cmf.trajectory_matrix({x: 0, y: -2}).limit_equivalent(
        cmf.M(y, False).walk({x: 0, y: -1}, 2, {x: x, y: y})
    )


def test_trajectory_matrix_negative():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    expected = (
        cmf.M(a, sign=True)
        * cmf.M(b, sign=False).subs({a: a + 1})
        * cmf.M(b, sign=False).subs({a: a + 1, b: b - 1})
        * cmf.M(c, sign=False).subs({a: a + 1, b: b - 2})
    )

    assert cmf.trajectory_matrix({a: 1, b: -2, c: -1}).limit_equivalent(expected)


def test_trajectory_matrix_variable_reduction():
    cmf = known_cmfs.e()
    trajectory = {x: -2, y: 3}
    start = {x: 5, y: -7}
    assert cmf.trajectory_matrix(trajectory).subs(
        CMF.variable_reduction_substitution(trajectory, start, n)
    ) == cmf.trajectory_matrix(trajectory, start)


def test_walk_axis():
    cmf = known_cmfs.e()
    start = {x: 1, y: 1}
    assert cmf.walk({x: 1, y: 0}, 17, start) == cmf.M(x).walk({x: 1, y: 0}, 17, start)
    assert cmf.walk({x: 0, y: 1}, 17, start) == cmf.M(y).walk({x: 0, y: 1}, 17, start)


def test_walk_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 1}
    start = {x: 1, y: 1}
    Mxy = cmf.trajectory_matrix(trajectory, start)
    assert cmf.walk(trajectory, 9, start) == Mxy.walk({n: 1}, 9, {n: 1})


def test_limit_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 1}
    start = {x: 1, y: 1}
    Mxy = cmf.trajectory_matrix(trajectory)
    assert cmf.limit(trajectory, 17, start) == Mxy.limit(trajectory, 17, start)


def test_limit_vectors():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 3}
    depths = [12, 13, 17]
    start = {x: 2, y: 1}
    p_vectors = [Matrix([[1, 2, 3]]), Matrix([4, 5, 6])]
    q_vectors = [Matrix([[4, 5, 6]]), Matrix([1, 2, 3])]
    expected = cmf.limit(trajectory, depths, start)
    for lim in expected:
        lim.p_vectors = p_vectors
        lim.q_vectors = q_vectors
    assert expected == cmf.limit(
        trajectory, depths, start, p_vectors=p_vectors, q_vectors=q_vectors
    )


def test_walk_list():
    cmf = known_cmfs.e()
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29]
    assert cmf.walk(trajectory, iterations, start) == [
        cmf.walk(trajectory, i, start) for i in iterations
    ]


def test_variable_reduction_substitution_axis():
    x_axis = {x: 1, y: 0}
    y_axis = {x: 0, y: 1}
    assert {x: n, y: 0} == CMF.variable_reduction_substitution(x_axis, x_axis, n)
    assert {x: 0, y: n} == CMF.variable_reduction_substitution(y_axis, y_axis, n)


def test_variable_reduction_substitution_diagonal():
    trajectory = {x: 1, y: 2}
    start = {x: 3, y: 5}
    assert {x: n + 2, y: 2 * n + 3} == CMF.variable_reduction_substitution(
        trajectory, start, n
    )


def test_N():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    assert 2 == cmf.N()


def test_dim():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    assert 3 == cmf.dim()


def test_variable_reduction_walk_equivalence():
    cmf = known_cmfs.e()
    iterations = 7
    trajectory = {x: 1, y: 1}
    start = {x: 3, y: 5}
    unsubbed = cmf.trajectory_matrix(trajectory)
    subbed = cmf.trajectory_matrix(trajectory, start)
    assert subbed.walk({n: 1}, iterations, {n: 1}) == unsubbed.walk(
        trajectory, iterations, start
    )


def test_delta_correctness():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    depth = 100
    trajectory = {a: 1, b: 0, c: 1}
    start = {a: 1, b: 1, c: 1}
    l1, l2 = cmf.limit(trajectory, [depth, 2 * depth], start)
    assert cmf.delta(trajectory, depth, start) == l1.delta(l2.as_float())
    assert cmf.delta(trajectory, depth, start, limit=l2.as_float()) == l1.delta(
        l2.as_float()
    )


def test_blind_delta_sequence_agrees_with_blind_delta():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    depth = 30
    trajectory = {a: 1, b: 2, c: 0}
    start = {a: 2, b: 3, c: 4}
    limit = cmf.limit(trajectory, 2 * depth, start).as_float()
    delta_sequence = cmf.delta_sequence(trajectory, depth, start)
    sequence_of_deltas = [
        cmf.delta(trajectory, i, start, limit=limit) for i in range(1, depth + 1)
    ]
    assert delta_sequence == sequence_of_deltas
    assert len(delta_sequence) == depth
