from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sp
from flint import fmpq_mat, fmpq, fmpz_mat

import ramanujantools as rt
from ramanujantools import Position
from ramanujantools.utils import batched, Batchable


class NumericMatrix(fmpq_mat):
    @staticmethod
    def _fmpz_eye(size: int) -> fmpz_mat:
        """
        Returns an identity matrix over FLINT integers.
        """
        return fmpz_mat(
            [
                [1 if row_index == col_index else 0 for col_index in range(size)]
                for row_index in range(size)
            ]
        )

    @staticmethod
    def _to_fmpq(value: sp.Expr | int) -> fmpq:
        """
        Converts an exact integer or rational SymPy value into a FLINT rational.
        """
        value = sp.S(value)
        if isinstance(value, sp.Integer):
            return fmpq(int(value))
        if isinstance(value, sp.Rational):
            return fmpq(int(value.p), int(value.q))
        raise TypeError(f"Expected an exact rational value, got {value}")

    @staticmethod
    def _compile_batched_walk_matrix(
        matrix: rt.Matrix, symbol: sp.Symbol
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compiles a one-symbol rational matrix into integer coefficient tensors.

        The common denominator is used as one scalar per step, while the inflated
        polynomial entries are evaluated together from their coefficient columns.
        """
        cache = getattr(matrix, "_numeric_batched_walk_cache", None)
        if cache is None:
            cache = {}
            matrix._numeric_batched_walk_cache = cache
        if symbol in cache:
            return cache[symbol]

        common_denominator = sp.factor(
            sp.lcm_list(
                [
                    sp.denom(matrix[row_index, col_index])
                    for row_index in range(matrix.rows)
                    for col_index in range(matrix.cols)
                ]
            )
        )

        denominator_poly = sp.Poly(common_denominator, symbol)
        flattened_denominator_coefficients = np.zeros(1, dtype=object)
        flattened_denominator_coefficients[0] = sp.Integer(0)

        coefficients = []
        max_degree = 0
        for row_index in range(matrix.rows):
            row_coefficients = []
            for col_index in range(matrix.cols):
                polynomial = sp.Poly(matrix[row_index, col_index] * common_denominator, symbol)
                if not all(coefficient.is_integer for coefficient in polynomial.all_coeffs()):
                    raise sp.PolynomialError(
                        f"Inflated entry does not have integer coefficients: {matrix[row_index, col_index] * common_denominator}"
                    )
                entry_coefficients = tuple(
                    int(coefficient) for coefficient in reversed(polynomial.all_coeffs())
                ) or (0,)
                max_degree = max(max_degree, len(entry_coefficients) - 1)
                row_coefficients.append(entry_coefficients)
            coefficients.append(tuple(row_coefficients))

        flattened_entry_coefficients = np.zeros(
            (max_degree + 1, matrix.rows * matrix.cols), dtype=object
        )
        for row_index in range(matrix.rows):
            for col_index in range(matrix.cols):
                entry_index = row_index * matrix.cols + col_index
                for degree, coefficient in enumerate(coefficients[row_index][col_index]):
                    flattened_entry_coefficients[degree, entry_index] = coefficient

        flattened_denominator_coefficients = np.zeros(max_degree + 1, dtype=object)
        if not all(coefficient.is_integer for coefficient in denominator_poly.all_coeffs()):
            raise sp.PolynomialError(
                f"Common denominator does not have integer coefficients: {common_denominator}"
            )
        for degree, coefficient in enumerate(reversed(denominator_poly.all_coeffs())):
            flattened_denominator_coefficients[degree] = int(coefficient)
        compiled = flattened_denominator_coefficients, flattened_entry_coefficients, max_degree
        cache[symbol] = compiled
        return compiled

    @staticmethod
    def _can_use_batched_evaluation(
        matrix: rt.Matrix, trajectory: Position, start: Position
    ) -> bool:
        """
        Returns True when the walk can use one-symbol batched polynomial evaluation.
        """
        if len(trajectory) != 1 or len(start) != 1:
            return False

        symbol = next(iter(trajectory.keys()))
        if not sp.S(trajectory[symbol]).is_integer or not sp.S(start[symbol]).is_integer:
            return False
        if set(matrix.free_symbols) - {symbol}:
            return False

        try:
            NumericMatrix._compile_batched_walk_matrix(matrix, symbol)
        except sp.PolynomialError:
            return False
        return True

    @staticmethod
    def _batched_step_matrices(
        matrix: rt.Matrix,
        trajectory: Position,
        depth: int,
        start: Position,
    ) -> list["NumericMatrix"]:
        """
        Generates all one-symbol step matrices with a Vandermonde powers table.
        """
        if depth == 0:
            return []

        symbol = next(iter(trajectory.keys()))
        step_size = int(sp.S(trajectory[symbol]))
        start_value = int(sp.S(start[symbol]))
        denominator_coefficients, flattened_entry_coefficients, max_degree = NumericMatrix._compile_batched_walk_matrix(
            matrix, symbol
        )

        evaluation_points = np.array(
            [start_value + offset * step_size for offset in range(depth)], dtype=object
        )
        vandermonde = np.empty((depth, max_degree + 1), dtype=object)
        vandermonde[:, 0] = 1
        for degree in range(1, max_degree + 1):
            vandermonde[:, degree] = vandermonde[:, degree - 1] * evaluation_points

        denominators = vandermonde @ denominator_coefficients
        evaluated_entries = vandermonde @ flattened_entry_coefficients

        step_matrices = []
        for depth_index in range(depth):
            denominator = int(denominators[depth_index])
            if denominator == 0:
                raise ZeroDivisionError(
                    f"Common denominator vanished at {symbol}={evaluation_points[depth_index]}"
                )
            rows = []
            for row_index in range(matrix.rows):
                row_values = []
                for col_index in range(matrix.cols):
                    entry_index = row_index * matrix.cols + col_index
                    numerator = int(evaluated_entries[depth_index, entry_index])
                    row_values.append(fmpq(numerator, denominator))
                rows.append(row_values)
            step_matrices.append(NumericMatrix(rows))
        return step_matrices

    @staticmethod
    def _batched_integer_step_data(
        matrix: rt.Matrix,
        trajectory: Position,
        depth: int,
        start: Position,
    ) -> tuple[list[fmpz_mat], list[int]]:
        """
        Generates inflated integer step matrices together with their scalar denominators.
        """
        if depth == 0:
            return [], []

        symbol = next(iter(trajectory.keys()))
        step_size = int(sp.S(trajectory[symbol]))
        start_value = int(sp.S(start[symbol]))
        denominator_coefficients, flattened_entry_coefficients, max_degree = NumericMatrix._compile_batched_walk_matrix(
            matrix, symbol
        )

        evaluation_points = np.array(
            [start_value + offset * step_size for offset in range(depth)], dtype=object
        )
        vandermonde = np.empty((depth, max_degree + 1), dtype=object)
        vandermonde[:, 0] = 1
        for degree in range(1, max_degree + 1):
            vandermonde[:, degree] = vandermonde[:, degree - 1] * evaluation_points

        denominators = [int(value) for value in (vandermonde @ denominator_coefficients)]
        evaluated_entries = vandermonde @ flattened_entry_coefficients

        step_matrices = []
        for depth_index in range(depth):
            if denominators[depth_index] == 0:
                raise ZeroDivisionError(
                    f"Common denominator vanished at {symbol}={evaluation_points[depth_index]}"
                )
            rows = []
            for row_index in range(matrix.rows):
                row_values = []
                for col_index in range(matrix.cols):
                    entry_index = row_index * matrix.cols + col_index
                    row_values.append(int(evaluated_entries[depth_index, entry_index]))
                rows.append(row_values)
            step_matrices.append(fmpz_mat(rows))
        return step_matrices, denominators

    @staticmethod
    def _numeric_from_integer_product(product: fmpz_mat, scalar: int) -> "NumericMatrix":
        """
        Converts one exact inflated integer product back into a FLINT rational matrix.
        """
        if scalar == 0:
            raise ZeroDivisionError("Cannot recover a rational walk from a zero scalar")
        return NumericMatrix(
            [
                [fmpq(int(product[row_index, col_index]), scalar) for col_index in range(product.ncols())]
                for row_index in range(product.nrows())
            ]
        )

    @staticmethod
    def _batched_integer_walk(
        matrix: rt.Matrix,
        iterations: Batchable[int],
        trajectory: Position,
        start: Position,
    ) -> Batchable["NumericMatrix"]:
        """
        Runs the reduced walk through inflated integer FLINT matrices and divides only at checkpoints.
        """
        if not iterations:
            return []

        step_matrices, step_scalars = NumericMatrix._batched_integer_step_data(
            matrix, trajectory, iterations[-1], start
        )
        dim = matrix.rows

        sequential_threshold = 8

        def _product_tree(first: int, last: int) -> tuple[fmpz_mat, int]:
            span = last - first
            if span == 0:
                return step_matrices[first], step_scalars[first]
            if span <= sequential_threshold:
                result_matrix = step_matrices[first]
                result_scalar = step_scalars[first]
                for index in range(first + 1, last + 1):
                    result_matrix = result_matrix * step_matrices[index]
                    result_scalar *= step_scalars[index]
                return result_matrix, result_scalar
            midpoint = (first + last) >> 1
            left_matrix, left_scalar = _product_tree(first, midpoint)
            right_matrix, right_scalar = _product_tree(midpoint + 1, last)
            return left_matrix * right_matrix, left_scalar * right_scalar

        accumulated_matrix = NumericMatrix._fmpz_eye(dim)
        accumulated_scalar = 1
        results = []
        previous_depth = 0
        for target_depth in iterations:
            if target_depth > previous_depth:
                segment_matrix, segment_scalar = _product_tree(previous_depth, target_depth - 1)
                accumulated_matrix = accumulated_matrix * segment_matrix
                accumulated_scalar *= segment_scalar
            results.append(
                NumericMatrix._numeric_from_integer_product(
                    accumulated_matrix, accumulated_scalar
                )
            )
            previous_depth = target_depth
        return results

    @staticmethod
    def eye(N: int):
        """
        Returns an identity matrix of size N
        """
        retval = NumericMatrix(N, N)
        for i in range(N):
            retval[i, i] = 1
        return retval

    @staticmethod
    def lambda_from_rt(matrix: rt.Matrix) -> Callable:
        """
        Returns a function that evaluates the matrix at given a point
        and returns it as a NumericMatrix.
        """

        def _mul(args, subs):
            result = fmpq(1)
            for arg in args:
                val = arg(subs) if callable(arg) and subs is not None else arg
                result *= val
            return result

        def flint_lambdify(expr):
            """
            A recursive walker that constructs a flint based lambda expression
            from the sympy expression.
            """
            if isinstance(expr, sp.Integer):
                return fmpq(int(expr))
            elif isinstance(expr, sp.Rational):
                return fmpq(expr.p, expr.q)
            elif isinstance(expr, sp.Symbol):
                # Return a function that takes substitutions as a dict
                return lambda subs: fmpq(subs[expr])
            elif isinstance(expr, sp.Add):
                args = [flint_lambdify(arg) for arg in expr.args]
                if any(callable(arg) for arg in args):
                    return lambda subs: sum(
                        arg(subs) if callable(arg) else arg for arg in args
                    )
                else:
                    return sum(args)
            elif isinstance(expr, sp.Mul):
                args = [flint_lambdify(arg) for arg in expr.args]
                if any(callable(arg) for arg in args):
                    return lambda subs: _mul(args, subs)
                else:
                    return _mul(args, None)
            elif isinstance(expr, sp.Pow):
                base = flint_lambdify(expr.base)
                exp = flint_lambdify(expr.exp)
                if callable(base) or callable(exp):
                    return lambda subs: (base(subs) if callable(base) else base) ** int(
                        exp(subs) if callable(exp) else exp
                    )
                else:
                    return base ** int(exp)
            elif isinstance(expr, sp.MatrixBase):
                # Recursively convert each element
                rows, cols = expr.shape
                data = [
                    [flint_lambdify(expr[i, j]) for j in range(cols)]
                    for i in range(rows)
                ]

                return lambda subs: NumericMatrix(
                    [
                        [cell(subs) if callable(cell) else cell for cell in row]
                        for row in data
                    ]
                )

            else:
                raise NotImplementedError(f"Unsupported sympy type: {type(expr)}")

        return flint_lambdify(matrix)

    @staticmethod
    @batched("iterations")
    def walk(
        matrix: rt.Matrix,
        trajectory: Position,
        iterations: Batchable[int],
        start: Position,
    ) -> Batchable[NumericMatrix]:
        if NumericMatrix._can_use_batched_evaluation(matrix, trajectory, start):
            return NumericMatrix._batched_integer_walk(
                matrix, iterations, trajectory, start
            )

        N = iterations[-1]
        dim = matrix.rows
        fast_subs = NumericMatrix.lambda_from_rt(matrix)
        position = start.copy()
        step_matrices = []
        for _ in range(N):
            step_matrices.append(fast_subs(position))
            position += trajectory

        # Below this range size, a sequential loop is faster than recursion.
        SEQUENTIAL_THRESHOLD = 8

        def _product_tree(first, last):
            """Return step_matrices[first] * step_matrices[first+1] * … * step_matrices[last].

            Uses balanced divide-and-conquer so that intermediate products stay
            small, which is critical for FLINT rational matrices whose entry
            sizes grow with each multiplication.
            """
            span = last - first
            if span == 0:
                return step_matrices[first]
            if span <= SEQUENTIAL_THRESHOLD:
                result = step_matrices[first]
                for i in range(first + 1, last + 1):
                    result = result * step_matrices[i]
                return result
            mid = (first + last) >> 1
            return _product_tree(first, mid) * _product_tree(mid + 1, last)

        # Build results at each requested checkpoint.
        results = []
        accumulated = NumericMatrix.eye(dim)
        prev_depth = 0
        for target in iterations:
            if target > prev_depth:
                segment = _product_tree(prev_depth, target - 1)
                accumulated = accumulated * segment
            results.append(NumericMatrix(accumulated))
            prev_depth = target
        return results

    def to_rt(self) -> rt.Matrix:
        return rt.Matrix(self.nrows(), self.ncols(), list(self))

    def __neg__(self) -> NumericMatrix:
        return NumericMatrix(super().__neg__())

    def __add__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__add__(other))

    def __radd__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__radd__(other))

    def __sub__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__sub__(other))

    def __rsub__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__rsub__(other))

    def __mul__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__mul__(other))

    def __rmul__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__rmul__(other))

    def __div__(self, other) -> NumericMatrix:
        return NumericMatrix(super().__div__(other))
