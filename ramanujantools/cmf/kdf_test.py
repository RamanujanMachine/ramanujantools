import pickle

import pytest
import sympy as sp
import mpmath as mp
from sympy.abc import n, x, y

from ramanujantools import Position
from ramanujantools.cmf import CMF, KdF

a0, a1, b0, nd0, ne0 = sp.symbols("a0 a1 b0 nd0 ne0")


def test_rank_and_axes():
    cmf = KdF(2, 1, 0, 1, 1, 0)
    assert 4 == cmf.rank() == KdF.predict_rank(2, 1, 0, 1, 1, 0)
    assert {a0, a1, b0, nd0, ne0} == cmf.axes()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        KdF(1, 0, 0, -1, 0, 0)
    with pytest.raises(ValueError):
        KdF(1, 0, 0, 1, 0, 0, 0.5, 0.25)


def test_degenerate_arguments_generic_raises():
    with pytest.raises(ValueError):
        KdF(1, 1, 0, 1, 0, 0, 1, 1, method="generic")


def test_degenerate_arguments_fall_back_to_specialized():
    """At x=y=1 the generic basis degenerates, and the contiguity module of the function is used."""
    cmf = KdF(2, 1, 0, 1, 1, 0, 1, 1)
    assert "specialized" == cmf.basis_info["method"]
    assert 3 == cmf.rank() < KdF.predict_rank(2, 1, 0, 1, 1, 0)
    cmf.validate_conserving()


def test_singular_function_raises():
    """F^{0:1;0}(x, y) = (1 - x)^(-b) has no finite nonzero module at x=1."""
    with pytest.raises(ValueError):
        KdF(0, 1, 0, 0, 0, 0, 1, 1)


def test_conserving():
    KdF(1, 1, 0, 1, 1, 0).validate_conserving()
    KdF(1, 0, 0, 1, 1, 0, sp.Rational(1, 7), sp.Rational(1, 11)).validate_conserving()


def test_subs():
    assert KdF(1, 0, 0, 1, 1, 0, 2, 3) == KdF(1, 0, 0, 1, 1, 0).subs({x: 2, y: 3})


def test_pickle():
    cmf = KdF(2, 1, 0, 1, 1, 0, 1, 1)
    cmf.determinant(a0)
    unpickled = pickle.loads(pickle.dumps(cmf))
    assert list(cmf.matrices.keys()) == list(unpickled.matrices.keys())
    assert cmf == unpickled
    assert cmf._determinants == unpickled._determinants


@pytest.mark.parametrize(
    "counts, arguments, start",
    [
        (
            (2, 1, 0, 1, 1, 0),
            (sp.Rational(1, 7), sp.Rational(1, 11)),
            (3, 2, 5, -4, -2),
        ),
        ((2, 1, 0, 1, 1, 0), (1, 1), (sp.Rational(1, 2), 2, sp.Rational(3, 4), -3, -2)),
        ((1, 1, 0, 1, 1, 0), (x, y), (sp.Rational(1, 3), 2, -4, -3)),
    ],
)
def test_trajectory_matrix_native_steps(counts, arguments, start):
    """The FLINT step matrices (with backward steps) agree with the default sympy path on the same matrices."""
    cmf = KdF(*counts, *arguments)
    axes = list(cmf.matrices.keys())
    start = Position(dict(zip(axes, start)))
    trajectory = Position(
        {axis: (-1) ** k * (k % 3 + 1) for k, axis in enumerate(axes)}
    )
    reference = CMF(dict(cmf.matrices), validate=False)
    assert reference.trajectory_matrix(trajectory, start) == cmf.trajectory_matrix(
        trajectory, start
    )


def test_step_matrix_poles_match_default():
    """On singular hyperplanes (a0 + nd0 + 1 = 0, nd0 = -1) the native step matrices have the same poles."""
    cmf = KdF(2, 1, 0, 1, 1, 0, 1, 1)
    axes = list(cmf.matrices.keys())
    for base in ({a0: sp.Rational(1, 2) + n, nd0: sp.Rational(-3, 2) - n}, {nd0: -1}):
        position = Position(
            {axis: sp.Rational(k + 2, 7) for k, axis in enumerate(axes)}
        )
        position.update(base)
        ctx = cmf.ctx(position)
        for axis in axes:
            for sign in (True, False):
                results = []
                for step_matrix in (KdF._step_matrix, CMF._step_matrix):
                    try:
                        results.append(
                            step_matrix(cmf, axis, sign, position, ctx).factor()
                        )
                    except ZeroDivisionError:
                        results.append(None)
                assert results[0] == results[1]


def test_contiguous_relations_of_series():
    """
    F^{1:0;0}_{1:0;0}(x, y) = 1F1(a; d; x + y), with basis (1, theta_x), is checked on both an upper (a)
    and a lower (nd = -d) axis against the series itself.
    """
    xv, yv = sp.Rational(1, 3), sp.Rational(1, 5)
    cmf = KdF(1, 0, 0, 1, 0, 0, xv, yv)
    a, d = sp.Rational(3, 2), sp.Rational(7, 3)

    def state(a, d):
        t = mp.mpf(xv + yv)
        return mp.matrix(
            [[mp.hyp1f1(a, d, t), t * mp.diff(lambda s: mp.hyp1f1(a, d, s), t)]]
        )

    def matrix(axis):
        return mp.matrix(cmf.M(axis).subs({a0: a, nd0: -d}).evalf(30).tolist())

    with mp.workdps(30):
        start = state(a, d)
        assert mp.norm(state(a + 1, d) - start * matrix(a0)) < 1e-20
        assert mp.norm(state(a, d - 1) - start * matrix(nd0)) < 1e-20


@pytest.mark.parametrize(
    "counts, arguments, axis",
    [
        ((1, 0, 0, 1, 1, 0), (sp.Rational(1, 7), sp.Rational(1, 11)), a0),
        ((1, 0, 0, 1, 1, 0), (sp.Rational(1, 7), sp.Rational(1, 11)), nd0),
        ((2, 1, 0, 1, 1, 0), (sp.Rational(1, 7), sp.Rational(1, 11)), b0),
        ((2, 1, 0, 1, 1, 0), (sp.Rational(1, 7), sp.Rational(1, 11)), a1),
        ((2, 1, 0, 1, 1, 0), (1, 1), a1),
        ((2, 1, 0, 1, 1, 0), (1, 1), ne0),
    ],
)
def test_determinant(counts, arguments, axis):
    """Tests KdF.determinant against the actual matrix determinant (generic and specialized constructions)."""
    cmf = KdF(*counts, *arguments)
    assert sp.factor(cmf.matrices[axis].det()) == cmf.determinant(axis)


def test_determinant_after_pickle():
    cmf = KdF(2, 1, 0, 1, 1, 0, sp.Rational(1, 7), sp.Rational(1, 11))
    unpickled = pickle.loads(pickle.dumps(cmf))
    for axis in cmf.axes():
        assert cmf.determinant(axis) == unpickled.determinant(axis)


def test_determinants_in_parallel():
    cmf = KdF(2, 1, 0, 1, 1, 0, 1, 1)
    expected = {axis: cmf.determinant(axis) for axis in cmf.axes()}
    cmf._determinants.clear()
    assert expected == cmf.determinants(processes=2)


def test_determinant_invalid_axis():
    with pytest.raises(ValueError):
        KdF(1, 0, 0, 1, 1, 0).determinant(sp.Symbol("z0"))


@pytest.mark.parametrize(
    "counts, arguments",
    [
        ((2, 1, 0, 1, 1, 0), (sp.Rational(1, 7), sp.Rational(1, 11))),
        ((2, 1, 0, 1, 1, 0), (1, 1)),
        ((1, 1, 1, 1, 1, 0), (1, 1)),
        ((1, 1, 0, 1, 1, 0), (x, y)),
    ],
)
def test_poles(counts, arguments):
    """Forward poles are exactly the factors of the reduced entry denominators; backward poles contain those of
    the inverse."""
    cmf = KdF(*counts, *arguments)
    poles = cmf.poles()

    def denominator_factors(matrix):
        return {
            sp.factor(f).as_coeff_Mul()[1]
            for entry in matrix
            for f, _ in sp.factor_list(sp.fraction(sp.cancel(entry))[1])[1]
            if f.free_symbols
        }

    for axis in cmf.axes():
        assert denominator_factors(cmf.M(axis)) == set(poles[axis]["forward"])
        assert denominator_factors(cmf.M(axis, False)) <= set(poles[axis]["backward"])


def test_has_nonlinear_poles():
    assert not KdF(2, 1, 0, 1, 1, 0, 1, 1).has_nonlinear_poles()
    assert KdF(
        0, 0, 0, 2, 1, 1, sp.Rational(1, 4), sp.Rational(1, 4)
    ).has_nonlinear_poles()


@pytest.mark.parametrize(
    "counts, arguments",
    [
        ((2, 1, 0, 1, 1, 0), (sp.Rational(1, 7), sp.Rational(1, 11))),
        ((2, 1, 0, 1, 1, 0), (1, 1)),
        ((1, 1, 0, 1, 1, 0), (x, y)),
    ],
)
def test_negative_matrix(counts, arguments):
    """M(axis, sign=False) (FLINT, per block by symmetry) equals the sympy inverse at axis - 1."""
    cmf = KdF(*counts, *arguments)
    for axis in cmf.axes():
        expected = cmf.M(axis).inverse().subs({axis: axis - 1})
        assert sp.simplify(cmf.M(axis, False) - expected) == sp.zeros(cmf.rank())
