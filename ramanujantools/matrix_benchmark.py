"""Benchmarks for Matrix.walk().

Run with: pytest matrix_benchmark.py --benchmark-only
"""

import pytest
from sympy.abc import n

from ramanujantools import Matrix

MATRICES = {
    "2x2_pcf": Matrix([[0, -(n**2)], [1, n + 1]]),
    "2x2_rational": Matrix([[n**3 - 1, -(n**2)], [n**2 + 17, 1 / (n + 1)]]),
    "3x3_rational": Matrix(
        [
            [n**3 - 1, -(n**2), 2],
            [n**2 + 17, n + 1, -(n**3)],
            [1 / (n + 2), 12 * n**2 + 13 * n + 14 * n, 19],
        ]
    ),
    "5x5_rational": Matrix(
        [
            [n**3 - 1, -(n**2), 2, n, 1],
            [n**2 + 17, n + 1, -(n**3), 7, n**3 - 2],
            [1 / (n + 2), 12 * n**2 + 13 * n + 14 * n, 19, 1 / n, n**2],
            [3 * n**3 - 2, n + 1, 1 / (n + 1), 7, n**2 - 1],
            [n, n**2, n**3, 1 - n, -n * (n - 7) + 3],
        ]
    ),
    "5x5_polynomial": Matrix(
        [
            [n**3 - 1, -(n**2), 2, n, 1],
            [n**2 + 17, n + 1, -(n**3), 7, n**3 - 2],
            [n + 2, 12 * n**2 + 13 * n + 14 * n, 19, 1, n**2],
            [3 * n**3 - 2, n + 1, n + 1, 7, n**2 - 1],
            [n, n**2, n**3, 1 - n, -n * (n - 7) + 3],
        ]
    ),
}

DEPTHS = [100, 500, 1000, 2000]


def walk_benchmark(matrix, trajectory, iterations, start):
    Matrix._walk_inner.cache_clear()
    return matrix.walk(trajectory, iterations, start)


@pytest.mark.parametrize("depth", DEPTHS, ids=[f"N={d}" for d in DEPTHS])
@pytest.mark.parametrize("matrix_name", MATRICES.keys())
def test_walk_benchmark(benchmark, matrix_name, depth):
    benchmark(walk_benchmark, MATRICES[matrix_name], {n: 1}, depth, {n: 1})


def test_as_companion_3x3_2f2_benchmark(benchmark):
    matrix = Matrix(
        [
            [n**2 * (n + 2), 2 * n**2 * (n + 1), n**3 * (n + 1)],
            [
                n**2 + 6 * n + 2,
                n**3 + 10 * n**2 + 10 * n + 2,
                6 * n**3 + 5 * n**2 - 6 * n - 4,
            ],
            [-2 * n - 1, -(n**2) + n + 1, (3 * n + 2) * (3 * n + 4)],
        ]
    )
    benchmark(Matrix.as_companion, matrix)
