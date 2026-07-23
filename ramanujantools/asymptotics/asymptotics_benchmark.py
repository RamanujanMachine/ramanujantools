"""Benchmarks for the asymptotics module.

These benchmarks reuse the representative recurrences and exact expectations
from the asymptotics tests. They are intentionally excluded from normal pytest
collection by the repository's ``*_test.py`` pattern.

Run from the repository root with::

    pytest ramanujantools/asymptotics/asymptotics_benchmark.py --benchmark-only
"""

import pytest
import sympy as sp
from sympy.abc import n
from sympy.core.cache import clear_cache

from ramanujantools import LinearRecurrence, Matrix, Position
from ramanujantools.asymptotics import Reducer, SeriesMatrix
from ramanujantools.cmf import MeijerG


BENCHMARK_ROUNDS = 3
SPLIT_PRECISION = 12
SPLIT_RAMIFICATION = 3
SPLIT_TARGET = 2


def _clear_reducer_caches() -> None:
    """Clear every cross-call cache used by a timed asymptotics reduction."""
    clear_cache()
    LinearRecurrence._get_reducer.cache_clear()
    LinearRecurrence._get_reducer_at_precision.cache_clear()
    Matrix.inverse.cache_clear()
    Matrix.simplify.cache_clear()


def _prepare_recurrence(recurrence: LinearRecurrence) -> LinearRecurrence:
    """Exclude immutable companion-matrix construction from every timed round."""
    recurrence.recurrence_matrix
    return recurrence


def _run_asymptotics(
    recurrence: LinearRecurrence, precision: int | None
) -> list[sp.Expr]:
    return recurrence.asymptotics(precision=precision)


def _meijer_recurrence(trajectory: Position) -> LinearRecurrence:
    cmf = MeijerG(3, 2, 2, 3, 1)
    a0, a1 = sp.symbols("a:2")
    b0, b1, b2 = sp.symbols("b:3")
    start = Position({a0: 0, a1: 0, b0: 0, b1: 0, b2: 0})
    matrix = cmf.trajectory_matrix(trajectory, start)
    return _prepare_recurrence(LinearRecurrence(matrix))


def _split_fixture() -> tuple[Matrix, list[Matrix]]:
    """Build a dense algebraic split resembling the Euler reduction."""
    omega = (-1 + sp.sqrt(3) * sp.I) / 2
    target = Matrix.diag(1, omega, sp.conjugate(omega))
    coefficients = (
        [2 * Matrix.eye(3)]
        + [Matrix.zeros(3, 3)] * (SPLIT_TARGET - 1)
        + [target]
    )
    coefficients.extend(
        Matrix(
            3,
            3,
            lambda row, col: (
                order + row + col
                if row == col
                else (row + 1) * (col + 2) + order * omega
            ),
        )
        for order in range(1, SPLIT_PRECISION - SPLIT_TARGET)
    )
    return target, coefficients


SPLIT_JORDAN_FORM, SPLIT_COEFFICIENTS = _split_fixture()


def _setup_split_benchmark():
    _clear_reducer_caches()
    coefficients = [coefficient.copy() for coefficient in SPLIT_COEFFICIENTS]
    reducer = Reducer(
        SeriesMatrix(
            coefficients,
            p=SPLIT_RAMIFICATION,
            precision=SPLIT_PRECISION,
        ),
        factorial_power=0,
        precision=SPLIT_PRECISION,
        p=SPLIT_RAMIFICATION,
    )
    return (reducer, SPLIT_TARGET, SPLIT_JORDAN_FORM), {}


def _run_split(reducer: Reducer, k_target: int, jordan_form: Matrix) -> Reducer:
    reducer.split(k_target, jordan_form)
    return reducer


@pytest.mark.benchmark(group="asymptotics-split")
def test_reducer_split_algebraic_benchmark(benchmark):
    reducer = benchmark.pedantic(
        _run_split,
        setup=_setup_split_benchmark,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=1,
    )

    for coefficient in reducer.M.coeffs[SPLIT_TARGET + 1:]:
        off_diagonal = coefficient - sp.diag(*coefficient.diagonal())
        assert off_diagonal.is_zero_matrix


@pytest.mark.benchmark(group="asymptotics-meijer")
def test_meijer_euler_trajectory_benchmark(benchmark):
    a0, a1 = sp.symbols("a:2")
    b0, b1, b2 = sp.symbols("b:3")
    trajectory = Position({a0: 0, a1: 0, b0: 1, b1: 1, b2: 1})
    recurrence = _meijer_recurrence(trajectory)
    expected = [
        n ** sp.Rational(16, 3)
        * sp.exp(
            -sp.I
            * n ** sp.Rational(1, 3)
            * (
                3 * sp.sqrt(3) * n ** sp.Rational(1, 3)
                + 3 * sp.I * n ** sp.Rational(1, 3)
                + sp.sqrt(3)
                - sp.I
            )
            / 2
        )
        * sp.factorial(n) ** 2,
        n ** sp.Rational(16, 3)
        * sp.exp(
            sp.I
            * n ** sp.Rational(1, 3)
            * (
                3 * sp.sqrt(3) * n ** sp.Rational(1, 3)
                - 3 * sp.I * n ** sp.Rational(1, 3)
                + sp.sqrt(3)
                + sp.I
            )
            / 2
        )
        * sp.factorial(n) ** 2,
        n ** sp.Rational(16, 3)
        * sp.exp(-(n ** sp.Rational(1, 3)) * (3 * n ** sp.Rational(1, 3) - 1))
        * sp.factorial(n) ** 2,
    ]

    actual = benchmark.pedantic(
        _run_asymptotics,
        setup=_clear_reducer_caches,
        args=(recurrence, 12),
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=1,
    )

    assert actual == expected
