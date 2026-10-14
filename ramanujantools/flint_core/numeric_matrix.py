from __future__ import annotations

from typing import Callable

import sympy as sp
from flint import fmpq_mat, fmpq, fmpq_poly

import ramanujantools as rt
from ramanujantools import Position
from ramanujantools.utils import batched, Batchable


class NumericMatrix(fmpq_mat):
    # Below this walk depth, converting the matrix for `walk_step_lambda` costs more than it saves.
    FAST_WALK_MIN_DEPTH = 32

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
    def walk_step_lambda(
        matrix: rt.Matrix, trajectory: Position, start: Position
    ) -> Callable:
        """
        Returns a function that evaluates the matrix at step k of a walk, i.e at `start + k * trajectory`,
        and returns it as a NumericMatrix. Equivalent to `lambda_from_rt`, but faster:
        each entry is converted once to a numerator and denominator `fmpq_poly` in k, evaluated by FLINT.
        Denominators are never reduced, so a step raises ZeroDivisionError exactly when `lambda_from_rt` does.
        Raises NotImplementedError for anything else, including positions that are not python ints.
        """
        if any(
            type(value) is not int for value in [*start.values(), *trajectory.values()]
        ):
            raise NotImplementedError("Positions must be python ints")
        subs = {
            key: fmpq_poly([value, trajectory.get(key, 0)])
            for key, value in start.items()
        }
        one = fmpq_poly([1])
        # Nested denominators vanish from the result, but still raise in `lambda_from_rt`
        guards = []

        def convert(expr) -> tuple[fmpq_poly, fmpq_poly]:
            if isinstance(expr, sp.Rational):
                return fmpq_poly([fmpq(int(expr.p), int(expr.q))]), one
            if isinstance(expr, sp.Symbol) and expr in subs:
                return subs[expr], one
            if isinstance(expr, (sp.Add, sp.Mul)):
                numerator, denominator = convert(expr.args[0])
                for arg_numerator, arg_denominator in map(convert, expr.args[1:]):
                    if expr.is_Add:
                        numerator = (
                            numerator * arg_denominator + arg_numerator * denominator
                        )
                    else:
                        numerator = numerator * arg_numerator
                    denominator = denominator * arg_denominator
                return numerator, denominator
            if isinstance(expr, sp.Pow) and expr.exp.is_Integer:
                numerator, denominator = convert(expr.base)
                exponent = int(expr.exp)
                if exponent >= 0:
                    return numerator**exponent, denominator**exponent
                if denominator != 1:
                    guards.append(denominator)
                return denominator ** (-exponent), numerator ** (-exponent)
            raise NotImplementedError(f"Unsupported expression: {expr}")

        entries = [convert(expr) for expr in matrix]
        rows, cols = matrix.shape

        def step(k: int) -> NumericMatrix:
            if any(guard(k) == 0 for guard in guards):
                raise ZeroDivisionError("Nested denominator vanishes")
            values = [
                numerator(k) / denominator(k) for numerator, denominator in entries
            ]
            return NumericMatrix(rows, cols, values)

        return step

    @staticmethod
    @batched("iterations")
    def walk(
        matrix: rt.Matrix,
        trajectory: Position,
        iterations: Batchable[int],
        start: Position,
    ) -> Batchable[NumericMatrix]:
        """
        Walks as described in `rt.Matrix.walk`. Steps are evaluated with `walk_step_lambda` when possible,
        and with `lambda_from_rt` for short walks or unsupported matrices.
        """
        N = iterations[-1]
        dim = matrix.rows

        # Pre-evaluate all per-step matrices into a flat list.
        step = None
        if N >= NumericMatrix.FAST_WALK_MIN_DEPTH:
            try:
                step = NumericMatrix.walk_step_lambda(matrix, trajectory, start)
            except NotImplementedError:
                pass
        if step is not None:
            step_matrices = [step(k) for k in range(N)]
        else:
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
