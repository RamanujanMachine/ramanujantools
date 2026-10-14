import math
import random

import pytest
import sympy as sp
from sympy.abc import n, x, y

from ramanujantools import Matrix, Position
from ramanujantools.flint_core import NumericMatrix


def test_conversion():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    for i in range(1, 10):
        assert m.subs({n: i}) == NumericMatrix.lambda_from_rt(m)({n: i}).to_rt()


def test_walk():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    assert (
        m.walk({n: 2}, 100, {n: 3})
        == NumericMatrix.walk(m, Position({n: 2}), 100, Position({n: 3})).to_rt()
    )


def test_walk_list():
    m = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    rt_walk_matrices = m.walk({n: 2}, list(range(1, 100)), {n: 3})
    numeric_walk_matrices = NumericMatrix.walk(
        m, Position({n: 2}), list(range(1, 100)), Position({n: 3})
    )
    for rt_walk_matrix, numeric_walk_matrix in zip(
        rt_walk_matrices, numeric_walk_matrices
    ):
        assert rt_walk_matrix == numeric_walk_matrix.to_rt()


def fast_and_reference(compute):
    """
    Runs `compute` with walks evaluated by `walk_step_lambda` and by `lambda_from_rt`,
    asserts both return the same result or raise the same exception type, and returns it.
    """
    results = []
    for min_depth in [0, math.inf]:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(NumericMatrix, "FAST_WALK_MIN_DEPTH", min_depth)
            Matrix._walk_inner.cache_clear()
            try:
                results.append(compute())
            except Exception as error:
                results.append(type(error))
    Matrix._walk_inner.cache_clear()
    assert results[0] == results[1]
    return results[0]


def walk_both(matrix, trajectory, iterations, start):
    trajectory, start = Position(trajectory), Position(start)
    NumericMatrix.walk_step_lambda(matrix, trajectory, start)  # Not a fallback
    return fast_and_reference(
        lambda: NumericMatrix.walk(matrix, trajectory, iterations, start)
    )


@pytest.mark.parametrize("seed", range(10))
def test_walk_step_lambda_random(seed):
    rng = random.Random(seed)
    dim, degree = rng.randint(2, 5), rng.randint(1, 40)

    def polynomial():
        return sum(rng.randint(-(10**6), 10**6) * n**i for i in range(degree + 1))

    entries = [
        polynomial() / rng.choice([1, n + rng.randint(1, 9)]) for _ in range(dim**2)
    ]
    start, step = rng.randint(-3, 5), rng.randint(1, 3)
    walk_both(Matrix(dim, dim, entries), {n: step}, [1, 50], {n: start})


def test_walk_step_lambda_multivariate():
    matrix = Matrix([[x * y + 1, (x - y) / (x + y + 3)], [x**2 / 2, 1 / (x + 1)]])
    walk_both(matrix, {x: 2, y: -1}, [5, 40], {x: 3, y: 100})


@pytest.mark.parametrize("iterations", [40, [0], [1], [0, 1], list(range(1, 100))])
def test_walk_step_lambda_iterations(iterations):
    matrix = Matrix([[0, -(n**3)], [1, n**2 + 1 / (n + 1)]])
    walk_both(matrix, {n: 1}, iterations, {n: 1})


@pytest.mark.parametrize(
    "entry, singular_step",
    [
        (1 / n, 3),
        ((n**2 - 1) / (n - 1), 4),
        (n * (n + 1) / (n + 3) ** 2, 0),
        (1 / (2 + 1 / n), 3),  # inner denominator only
        (1 / (1 + 1 / n), 2),  # outer denominator
    ],
)
def test_walk_step_lambda_singular_step(entry, singular_step):
    matrix = Matrix([[entry, 1], [1, n]])
    assert walk_both(matrix, {n: 1}, singular_step, {n: -3}) is not ZeroDivisionError
    assert walk_both(matrix, {n: 1}, singular_step + 1, {n: -3}) is ZeroDivisionError


@pytest.mark.parametrize(
    "entry, start",
    [(sp.sqrt(n), 1), (sp.sin(n), 1), (n, sp.Integer(1))],
)
def test_walk_step_lambda_fallback(entry, start):
    matrix = Matrix([[entry, 1], [1, n]])
    with pytest.raises(NotImplementedError):
        NumericMatrix.walk_step_lambda(matrix, Position({n: 1}), Position({n: start}))
    fast_and_reference(
        lambda: NumericMatrix.walk(matrix, Position({n: 1}), 40, Position({n: start}))
    )
