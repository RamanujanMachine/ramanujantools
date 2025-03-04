from __future__ import annotations

from typing import Callable, Dict, List

from flint import fmpq_mat, fmpq
from sympy.utilities.lambdify import lambdastr
from sympy.printing.pycode import SymPyPrinter

import ramanujantools as rt
from ramanujantools import Position


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
        # Makes sp.Rational(1, 2) print as "S(1)/2" rather than "1/2""
        SymPyPrinter()._default_settings["sympy_integers"] = True

        symbols = list(sorted(matrix.free_symbols, key=str))
        evaluation_string = (
            lambdastr(symbols, matrix, printer=SymPyPrinter)
            .replace("ImmutableDenseMatrix", "NumericMatrix")
            .replace("**S", "**")
            .replace("S", "fmpq")
        )

        def fast_subs(substitutions: Dict):
            values = [substitutions[symbol] for symbol in symbols]
            return eval(evaluation_string)(*values)

        return fast_subs

    @staticmethod
    def walk(
        matrix: rt.Matrix, trajectory: Position, iterations: int, start: Position
    ) -> NumericMatrix:
        return NumericMatrix.walk_list(matrix, trajectory, [iterations], start)[0]

    @staticmethod
    def walk_list(
        matrix: rt.Matrix, trajectory: Position, iterations: List[int], start: Position
    ) -> List[NumericMatrix]:
        results = []
        position = start.copy()
        fast_subs = NumericMatrix.lambda_from_rt(matrix)
        retval = NumericMatrix.eye(matrix.rows)
        for depth in range(0, iterations[-1]):
            if depth in iterations:
                results.append(NumericMatrix(retval))
            retval *= fast_subs(position)
            position += trajectory
        results.append(NumericMatrix(retval))  # Last matrix, for iterations[-1]
        return results

    def to_rt(self) -> rt.Matrix:
        return rt.Matrix(self.nrows(), self.ncols(), list(self))
