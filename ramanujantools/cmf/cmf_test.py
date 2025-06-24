from pytest import raises

import sympy as sp
from sympy.abc import a, b, c, x, y, n

from ramanujantools import Position, Matrix, simplify
from ramanujantools.cmf import CMF, known_cmfs

c0, c1, c2, c3 = sp.symbols("c:4")


def test_N():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    assert 2 == cmf.N()


def test_dim():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    assert 3 == cmf.dim()


def test_dual_conserving():
    cmf = known_cmfs.cmf2()
    dual_cmf = cmf.dual()
    dual_cmf.assert_conserving


def test_dual_inverible():
    cmf = known_cmfs.cmf1()
    assert cmf == cmf.dual().dual()


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
    expected_parameters = {c0, c1, c2, c3}
    assert expected_axes == cmf.axes()
    assert expected_parameters == cmf.parameters()
    assert set().union(expected_axes, expected_parameters) == cmf.free_symbols()


def test_subs():
    cmf = known_cmfs.var_root_cmf()
    substitution = Position({c0: 1, c1: 2 * c1 - 3})
    assert cmf.subs(substitution) == CMF(
        {axis: matrix.subs(substitution) for axis, matrix in cmf.matrices.items()}
    )


def test_subs_axes_throw():
    cmf = known_cmfs.e()
    substitution = Position({x: 2})
    with raises(ValueError):
        cmf.subs(substitution)


def test_subs_non_linear_shift_throws():
    cmf = known_cmfs.e()
    substitution = Position({x: x**2})
    with raises(ValueError):
        cmf.subs(substitution)


def test_trajectory_matrix_axis():
    cmf = known_cmfs.e()
    assert cmf.trajectory_matrix({x: 3, y: 0}, {x: x, y: y}).walk(
        {n: 1}, 1, {n: 0}
    ) == simplify(cmf.M(x).walk({x: 1, y: 0}, 3, {x: x, y: y}))
    assert cmf.trajectory_matrix({x: 0, y: 2}, {x: x, y: y}).walk(
        {n: 1}, 1, {n: 0}
    ) == simplify(cmf.M(y).walk({x: 0, y: 1}, 2, {x: x, y: y}))


def test_trajectory_matrix_diagonal():
    cmf = known_cmfs.e()
    trajectory = Position({x: 1, y: 1})
    start = Position({x: x, y: y})
    assert cmf.trajectory_matrix(trajectory, start) == simplify(
        (cmf.M(x) * cmf.M(y)({x: x + 1})).subs(
            cmf.trajectory_substitution(trajectory, start, n)
        )
    )


def test_back_negates_forward():
    cmf = known_cmfs.e()
    assert Matrix.eye(2) == (cmf.M(x, True) * cmf.M(x, False).subs({x: x + 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(x, False) * cmf.M(x, True).subs({x: x - 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(y, True) * cmf.M(y, False).subs({y: y + 1})).reduce()
    assert Matrix.eye(2) == (cmf.M(y, False) * cmf.M(y, True).subs({y: y - 1})).reduce()


def test_trajectory_matrix_negative_axis():
    cmf = known_cmfs.e()
    start = {x: x, y: y}
    assert cmf.trajectory_matrix({x: -3, y: 0}, start).equal_projectively(
        cmf.M(x, False).walk(
            {x: -1, y: 0}, 3, cmf.trajectory_substitution({x: -3, y: 0}, start, n)
        )
    )
    assert cmf.trajectory_matrix({x: 0, y: -2}, start).equal_projectively(
        cmf.M(y, False).walk(
            {x: 0, y: -1}, 2, cmf.trajectory_substitution({x: 0, y: -2}, start, n)
        )
    )


def test_trajectory_matrix_negative():
    cmf = known_cmfs.hypergeometric_derived_2F1()
    expected = (
        cmf.M(a, sign=True)
        * cmf.M(b, sign=False).subs({a: a + 1})
        * cmf.M(b, sign=False).subs({a: a + 1, b: b - 1})
        * cmf.M(c, sign=False).subs({a: a + 1, b: b - 2})
    )
    actual = cmf.trajectory_matrix({a: 1, b: -2, c: -1}, {a: a, b: b, c: c})
    assert actual.walk({n: 1}, 1, {n: 0}).equal_projectively(expected)


def test_trajectory_matrix_rational():
    cmf = known_cmfs.e()
    start = Position({x: sp.Rational(2, 3), y: sp.Rational(1, 4)})
    trajectory = {x: 1, y: 1}
    symbolic_start = cmf.trajectory_substitution(trajectory, start, n)
    expected = (
        cmf.M(x).subs(symbolic_start) * cmf.M(y).subs(symbolic_start + Position({x: 1}))
    ).factor()
    actual = cmf.trajectory_matrix({x: 1, y: 1}, start)
    assert expected == actual


def test_walk_axis():
    cmf = known_cmfs.e()
    start = {x: 1, y: 1}
    assert cmf.walk({x: 1, y: 0}, 17, start) == cmf.M(x).walk({x: 1, y: 0}, 17, start)
    assert cmf.walk({x: 0, y: 1}, 17, start) == cmf.M(y).walk({x: 0, y: 1}, 17, start)


def test_walk_axis_multiplicity():
    cmf = known_cmfs.e()
    start = {x: 1, y: 1}
    trajectory = Position({x: 1, y: 0})
    depth = 4
    assert cmf.walk(depth * trajectory, 1, start) == cmf.M(x).walk(
        trajectory, depth, start
    )


def test_walk_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 1}
    start = {x: 1, y: 1}
    Mxy = cmf.trajectory_matrix(trajectory, start)
    assert cmf.walk(trajectory, 9, start) == Mxy.walk({n: 1}, 9, {n: 0})


def test_limit_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 1}
    start = {x: 1, y: 1}
    Mxy = cmf.walk(trajectory, 1, {x: x, y: y})
    assert cmf.limit(trajectory, 17, start) == Mxy.limit(trajectory, 17, start)


def test_limit_vectors():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 3}
    depths = [12, 13, 17]
    start = {x: 2, y: 1}
    initial_values = Matrix([[1, 2, 3], [4, 5, 6]])
    final_projection = Matrix([[1, 2], [3, 4], [5, 6]])
    expected = cmf.limit(trajectory, depths, start, initial_values, final_projection)
    for lim in expected:
        lim.initial_values = initial_values
        lim.final_projection = final_projection
    assert expected == cmf.limit(
        trajectory, depths, start, initial_values, final_projection
    )


def test_walk_list():
    cmf = known_cmfs.e()
    trajectory = {x: 2, y: 3}
    start = {x: 5, y: 7}
    iterations = [1, 2, 3, 17, 29]
    assert cmf.walk(trajectory, iterations, start) == [
        cmf.walk(trajectory, i, start) for i in iterations
    ]


def test_trajectory_substitution_throws_on_collision():
    cmf = known_cmfs.e()
    trajectory = Position({x: 1, y: 0})
    start = Position({x: n, y: 0})
    with raises(ValueError):
        cmf.trajectory_substitution(trajectory, start, n)
    assert {x: n + a, y: 0} == cmf.trajectory_substitution(trajectory, start, a)


def test_trajectory_substitution_axis():
    cmf = known_cmfs.e()
    x_axis = {x: 1, y: 0}
    y_axis = {x: 0, y: 1}
    origin = {x: 0, y: 0}
    assert {x: n, y: 0} == cmf.trajectory_substitution(x_axis, origin, n)
    assert {x: 0, y: n} == cmf.trajectory_substitution(y_axis, origin, n)


def test_trajectory_substitution_diagonal():
    cmf = known_cmfs.e()
    trajectory = {x: 1, y: 2}
    start = {x: 3, y: 5}
    assert {x: 3 + n, y: 5 + 2 * n} == cmf.trajectory_substitution(trajectory, start, n)


def test_trajectory_matrix_walk_equivalence():
    cmf = known_cmfs.e()
    iterations = 7
    trajectory = {x: 1, y: 1}
    start = {x: 3, y: 5}
    trajectory_matrix = cmf.trajectory_matrix(trajectory, start)
    assert trajectory_matrix.walk({n: 1}, iterations, {n: 0}) == cmf.walk(
        trajectory, iterations, start
    )


def test_work_non_integer_trajectory_throws():
    cmf = known_cmfs.e()
    with raises(ValueError):
        cmf.work(Position({x: 1, y: 1}), Position({x: 2, y: sp.Rational(5, 2)}))
    with raises(ValueError):
        cmf.work(Position({x: 1, y: n}), Position({x: 1, y: 2 * n}))


def test_work_numeric():
    cmf = known_cmfs.e()
    start = Position({x: 1, y: -1})
    end = Position({x: 17, y: -13})
    trajectory = end - start
    expected = cmf.walk(trajectory, 1, start)
    actual = cmf.work(start, end)
    assert expected == actual


def test_work_symbolic():
    cmf = known_cmfs.e()
    start = Position({x: 1 + x, y: -1 + 7 * y / 2})
    end = Position({x: 17 + x, y: -13 + 7 * y / 2})
    trajectory = end - start
    expected = cmf.walk(trajectory, 1, start)
    actual = cmf.work(start, end)
    print(expected)
    print(actual)
    assert expected == actual


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
