"""Benchmarks for Matrix.walk().

Run with: pytest matrix_benchmark.py --benchmark-only
"""

from sympy.abc import n

from ramanujantools import Matrix


# ---------------------------------------------------------------------------
# Shared test matrices
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# pytest-benchmark tests (through the public Matrix API)
# ---------------------------------------------------------------------------

def walk_benchmark(matrix, trajectory, iterations, start):
    Matrix._walk_inner.cache_clear()
    return matrix.walk(trajectory, iterations, start)


def test_walk_pcf_single_parameter_benchmark(benchmark):
    benchmark(walk_benchmark, MATRICES["2x2_pcf"], {n: 1}, 1000, {n: 1})


def test_walk_2x2_single_parameter_benchmark(benchmark):
    benchmark(walk_benchmark, MATRICES["2x2_rational"], {n: 1}, 1000, {n: 1})


def test_walk_3x3_single_parameter_benchmark(benchmark):
    benchmark(walk_benchmark, MATRICES["3x3_rational"], {n: 1}, 1000, {n: 1})


def test_walk_5x5_single_parameter_benchmark(benchmark):
    benchmark(walk_benchmark, MATRICES["5x5_rational"], {n: 1}, 1000, {n: 1})


def test_walk_5x5_polynomial_single_parameter_benchmark(benchmark):
    benchmark(walk_benchmark, MATRICES["5x5_polynomial"], {n: 1}, 1000, {n: 1})


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
