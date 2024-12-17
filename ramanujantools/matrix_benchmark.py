from sympy.abc import n

from ramanujantools import Matrix


def test_walk_2x2_single_parameter_benchmark(benchmark):
    matrix = Matrix([[0, -(n**2)], [1, n + 1]])
    benchmark(Matrix.walk, matrix, {n: 1}, 1000, {n: 1})


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
    benchmark(Matrix.as_companion, matrix, inflate_all=False)
