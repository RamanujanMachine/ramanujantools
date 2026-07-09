import sympy as sp
from sympy.abc import n, x, y

from ramanujantools import Matrix, Position
from ramanujantools.cmf import pFq
from ramanujantools.flint_core import NumericMatrix


def _manual_numeric_walk(
    matrix: Matrix, trajectory: Position, iterations: list[int], start: Position
) -> list[Matrix]:
    """
    Computes checkpoint walk products directly from per-step substitutions.
    """
    evaluator = NumericMatrix.lambda_from_rt(matrix)
    position = start.copy()
    step_matrices = []
    for _ in range(iterations[-1]):
        step_matrices.append(evaluator(position))
        position += trajectory

    results = []
    accumulated = NumericMatrix.eye(matrix.rows)
    previous_depth = 0
    for target_depth in iterations:
        for step_index in range(previous_depth, target_depth):
            accumulated = accumulated * step_matrices[step_index]
        results.append(accumulated.to_rt())
        previous_depth = target_depth
    return results


def test_conversion():
    """Checks that the FLINT evaluator matches direct substitution entrywise."""

    matrix = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )

    for value in range(1, 10):
        assert matrix.subs({n: value}) == NumericMatrix.lambda_from_rt(matrix)({n: value}).to_rt()


def test_walk_matches_manual_products():
    """Checks the optimized single-symbol walk against manual exact products."""

    matrix = Matrix(
        [
            [(n - 1) * (n**2 - n + 1) / n**3, -1 / n**3],
            [1 / n**3, (n + 1) * (n**2 + n + 1) / n**3],
        ]
    )
    iterations = [1, 2, 5, 10, 25]
    expected = _manual_numeric_walk(matrix, Position({n: 2}), iterations, Position({n: 3}))
    actual = NumericMatrix.walk(matrix, Position({n: 2}), iterations, Position({n: 3}))
    assert [numeric_matrix.to_rt() for numeric_matrix in actual] == expected


def test_walk_preserves_sparse_checkpoint_requests():
    """Checks that checkpoint accumulation stays correct for non-consecutive depths."""

    matrix = Matrix(
        [
            [(n + 2) / (n + 1), (2 * n + 3) / (n + 1)],
            [-(n + 5) / (n + 2), (3 * n + 7) / (n + 2)],
        ]
    )
    iterations = [3, 11, 17]
    expected = _manual_numeric_walk(matrix, Position({n: 1}), iterations, Position({n: 1}))
    actual = NumericMatrix.walk(matrix, Position({n: 1}), iterations, Position({n: 1}))
    assert [numeric_matrix.to_rt() for numeric_matrix in actual] == expected


def test_walk_multisymbol_fallback_matches_manual_products():
    """Checks that unsupported multi-symbol walks still follow the old exact path."""

    matrix = Matrix(
        [
            [(x + y + 1) / (x + 1), (x - y) / (y + 2)],
            [(2 * x + y) / (x + 2), (x + 3 * y + 1) / (y + 1)],
        ]
    )
    trajectory = Position({x: 1, y: 2})
    start = Position({x: 2, y: 3})
    iterations = [1, 4, 8]
    expected = _manual_numeric_walk(matrix, trajectory, iterations, start)
    actual = NumericMatrix.walk(matrix, trajectory, iterations, start)
    assert [numeric_matrix.to_rt() for numeric_matrix in actual] == expected


def _reduced_pfq_matrix(p_value: int, q_value: int, z_value: sp.Expr) -> Matrix:
    """
    Builds the reduced all-ones `pFq` trajectory matrix used in numeric walk benchmarks.
    """
    cmf = pFq(p_value, q_value, z_value)
    x_axes = sp.symbols(f"x:{p_value}")
    y_axes = sp.symbols(f"y:{q_value}")
    start = {axis: 1 for axis in x_axes + y_axes}
    trajectory = {
        **{axis: 1 for axis in x_axes},
        **{axis: 2 for axis in y_axes},
    }
    return cmf.trajectory_matrix(trajectory, start)


def test_reduced_pfq_2f1_walk_matches_manual_products():
    """Checks the optimized reduced `2F1` walk against manual exact products."""

    matrix = _reduced_pfq_matrix(2, 1, -1)
    iterations = [1, 2, 5, 10, 25]
    expected = _manual_numeric_walk(matrix, Position({n: 1}), iterations, Position({n: 1}))
    actual = NumericMatrix.walk(matrix, Position({n: 1}), iterations, Position({n: 1}))
    assert [numeric_matrix.to_rt() for numeric_matrix in actual] == expected


def test_reduced_pfq_3f2_walk_matches_manual_products():
    """Checks the optimized reduced `3F2` walk against manual exact products."""

    matrix = _reduced_pfq_matrix(3, 2, sp.Rational(1, 4))
    iterations = [1, 2, 5, 10, 25]
    expected = _manual_numeric_walk(matrix, Position({n: 1}), iterations, Position({n: 1}))
    actual = NumericMatrix.walk(matrix, Position({n: 1}), iterations, Position({n: 1}))
    assert [numeric_matrix.to_rt() for numeric_matrix in actual] == expected
