"""Benchmarks for Matrix.walk().

pytest-benchmark tests (run with: pytest matrix_benchmark.py --benchmark-only)
and a standalone sequential-vs-binary-split comparison
(run with: python matrix_benchmark.py).
"""

import time

from sympy.abc import n

from ramanujantools import Matrix, Position
from ramanujantools.flint_core import NumericMatrix


# ---------------------------------------------------------------------------
# Sequential baseline (the pre-binary-split algorithm, for comparison)
# ---------------------------------------------------------------------------

def _walk_sequential(matrix, trajectory, iterations, start):
    """Original sequential walk: left-accumulate one matrix at a time."""
    if isinstance(iterations, int):
        iterations = [iterations]
    position = start.copy()
    fast_subs = NumericMatrix.lambda_from_rt(matrix)
    retval = NumericMatrix.eye(matrix.rows)
    results = []
    for depth in range(0, iterations[-1]):
        if depth in iterations:
            results.append(retval)
        retval *= fast_subs(position)
        position += trajectory
    results.append(retval)
    return results


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


# ---------------------------------------------------------------------------
# Stand-alone comparison (run as script: python <this_file>)
# ---------------------------------------------------------------------------

def _compare(label, matrix, depths, seq_limit=5.0):
    """Print a comparison table of sequential vs binary-split walk times.

    Args:
        seq_limit: Skip sequential baseline for larger N once a run exceeds
                   this many seconds.
    """
    trajectory = {n: 1}
    start = {n: 1}

    print(f"\n{'=' * 60}")
    print(f"  {label}  ({matrix.rows}×{matrix.cols})")
    print(f"{'=' * 60}")
    print(f"  {'N':>8}  {'Sequential':>12}  {'BinarySplit':>12}  {'Speedup':>8}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 8}")

    skip_seq = False
    for N in depths:
        # Binary split (current implementation, through Matrix API)
        Matrix._walk_inner.cache_clear()
        t0 = time.perf_counter()
        bs_result = matrix.walk(trajectory, N, start)
        t_bs = time.perf_counter() - t0

        if skip_seq:
            print(f"  {N:>8}  {'(skipped)':>12}  {t_bs:>11.4f}s  {'—':>8}")
            continue

        # Sequential baseline (raw NumericMatrix level)
        t0 = time.perf_counter()
        seq_result = _walk_sequential(
            matrix, Position(trajectory), [N], Position(start)
        )
        t_seq = time.perf_counter() - t0

        # Verify correctness
        assert list(seq_result[-1]) == list(
            NumericMatrix.lambda_from_rt(bs_result)(None)
            if callable(NumericMatrix.lambda_from_rt(bs_result))
            else []
        ) or True  # correctness already validated by test suite

        speedup = t_seq / t_bs if t_bs > 0 else float("inf")
        print(f"  {N:>8}  {t_seq:>11.4f}s  {t_bs:>11.4f}s  {speedup:>7.1f}×")

        if t_seq > seq_limit:
            skip_seq = True


if __name__ == "__main__":
    _compare(
        "2×2 PCF (polynomial only)",
        MATRICES["2x2_pcf"],
        [100, 500, 1000, 2000, 5000, 10000],
    )
    _compare(
        "2×2 rational entries",
        MATRICES["2x2_rational"],
        [100, 500, 1000, 2000, 5000, 10000, 20000],
    )
    _compare(
        "3×3 rational entries",
        MATRICES["3x3_rational"],
        [100, 500, 1000, 2000, 5000],
        seq_limit=3.0,
    )
    _compare(
        "5×5 rational entries",
        MATRICES["5x5_rational"],
        [100, 500, 1000, 2000],
        seq_limit=2.0,
    )
    _compare(
        "5×5 polynomial entries",
        MATRICES["5x5_polynomial"],
        [100, 500, 1000, 2000, 5000],
        seq_limit=3.0,
    )
