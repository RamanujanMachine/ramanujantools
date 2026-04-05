from __future__ import annotations

from typing import Callable

import sympy as sp
from flint import fmpq_mat, fmpq

import ramanujantools as rt
from ramanujantools import Position
from ramanujantools.utils import batched, Batchable


class NumericMatrix(fmpq_mat):
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
        N = iterations[-1]
        fast_subs = NumericMatrix.lambda_from_rt(matrix)
        dim = matrix.rows

        # Pre-evaluate all per-step matrices into a flat list.
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
