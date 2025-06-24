from __future__ import annotations

import sympy as sp

import ramanujantools as rt
from ramanujantools import Position
from ramanujantools.flint_core import FlintRational, FlintContext
from ramanujantools.utils import batched, Batchable


class SymbolicMatrix:
    """
    Represents a Matrix of FlintRationals.

    It's logic is limited compared to the main Matrix, as it's designed for bottlenecks.
    """

    def __init__(
        self, rows: int, cols: int, values: list[FlintRational], ctx: FlintContext
    ) -> SymbolicMatrix:
        self._rows = rows
        self._cols = cols
        self.values = values
        self.ctx = ctx

    @staticmethod
    def from_sympy(matrix: rt.Matrix, ctx: FlintContext) -> SymbolicMatrix:
        """
        Converts a Matrix to SymbolicMatrix.
        Args:
            matrix: The matrix as ramanujantools.Matrix
            ctx: The desired mpoly context (which also defines the supported variables)
        """
        values = [FlintRational.from_sympy(cell, ctx) for cell in matrix]
        return SymbolicMatrix(matrix.rows, matrix.cols, values, ctx)

    @staticmethod
    def eye(N: int, ctx: FlintContext) -> SymbolicMatrix:
        """
        Creates an identity matrix of size N.

        Args:
            N: The squared matrix dimension
            ctx: The desired mpoly context (which also defines the supported variables)
        """
        values = [FlintRational.from_sympy(sp.simplify(0), ctx)] * N**2
        current = 0
        while current < N**2:
            values[current] += 1
            current += N + 1
        return SymbolicMatrix(N, N, values, ctx)

    def __getitem__(self, key):
        """
        Returns an element of the matrix.
        Supports both matrix[row, col] and matrix[index] syntax
        """
        if isinstance(key, tuple):
            row = key[0]
            col = key[1]
            return self.values[row * self.cols() + col]
        else:
            return self.values[key]

    def __setitem__(self, key, value):
        """
        Returns an element of the matrix.
        Supports both matrix[row, col] and matrix[index] syntax
        """
        if isinstance(key, tuple):
            row = key[0]
            col = key[1]
            self.values[row * self.cols() + col] = value
        else:
            self.values[key] = value

    def __eq__(self, other: FlintRational):
        return self.values == other.values

    def rows(self):
        return self._rows

    def cols(self):
        return self._cols

    def shape(self):
        return (self.rows(), self.cols())

    def row(self, index: int) -> list[FlintRational]:
        row = []
        for i in range(self.cols()):
            row.append(self[index, i])
        return row

    def col(self, index: int) -> list[FlintRational]:
        col = []
        for i in range(self.rows()):
            col.append(self[i, index])
        return col

    def data(self) -> str:
        data = []
        for row_index in range(self.rows()):
            data.append(self.row(row_index))
        return data

    def __repr__(self) -> str:
        return f"SymbolicMatrix({self.data()})"

    def __str__(self) -> str:
        return f"SymbolicMatrix({self.data()})"

    def __mul__(self, other: SymbolicMatrix | int) -> SymbolicMatrix:
        """
        Multiplies self by another SymbolicMatrix or a scalar.
        """
        if isinstance(other, SymbolicMatrix):
            if self.cols() != other.rows():
                raise ValueError("Attempting to multiply")
            elements = []
            for row in range(self.rows()):
                for col in range(other.cols()):
                    current = 0
                    for k in range(self.cols()):
                        current += self[row, k] * other[k, col]
                    elements.append(current)
            return SymbolicMatrix(self.rows(), other.cols(), elements, self.ctx)

        else:
            return SymbolicMatrix(
                self.rows(),
                self.cols(),
                [value * other for value in self.values],
                self.ctx,
            )

    def __rmul__(self, other: SymbolicMatrix | int) -> SymbolicMatrix:
        """
        Multiplies a SymbolicMatrix or a scalar by self.
        """
        if isinstance(other, int):
            return self * other
        return other * self

    def __truediv__(self, other: int) -> SymbolicMatrix:
        """
        Divides self by a scalar
        """
        if isinstance(other, SymbolicMatrix):
            raise ValueError("Attempted to divide by matrix!")
        return SymbolicMatrix(
            self.rows(),
            self.cols(),
            [value / other for value in self.values],
            self.ctx,
        )

    def subs(self, substitutions: dict) -> SymbolicMatrix:
        """
        Substitutes symbols in the matrix.
        """
        return SymbolicMatrix(
            self.rows(),
            self.cols(),
            [value.subs(substitutions) for value in self.values],
            self.ctx,
        )

    def factor(self) -> rt.Matrix:
        """
        Factors all elements in the matrix.
        """
        values = [value.factor() for value in self.values]
        return rt.Matrix(self.rows(), self.cols(), values)

    @batched("iterations")
    def walk(
        self, trajectory: dict, iterations: Batchable[int], start: dict
    ) -> Batchable[SymbolicMatrix]:
        r"""
        Returns the multiplication result of walking in a certain trajectory.

        The `walk` operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=self`, `(t_0, ..., t_k)=trajectory`, `n=iterations` and `(s_0, ..., s_k)=start`.

        This is a generalization of the basic (and most common) case $\prod_{i=0}^{n-1}M(s+i)$,
        where M=self, n=iterations and s=start.

        Args:
            trajectory: the trajectory of a single step in the walk, as defined above.
            iterations: The amount of multiplications to perform. Can be an integer value or a list of values.
            start: the starting point of the matrix multiplication
        Returns:
            The walk multiplication matrix as defined above.
            If iterations is list, returns a list of matrices.
        Raises:
            ValueError: If `self` is not a square matrix,
                        if `start` and `trajectory` have different keys,
                        if `iterations` contains duplicate values
        """
        position = Position(start)
        trajectory = Position(trajectory)
        results = []
        matrix = SymbolicMatrix.eye(self.rows(), self.ctx)
        for depth in range(0, iterations[-1]):
            if depth in iterations:
                results.append(matrix)
            matrix *= self.subs(position)
            position += trajectory
        results.append(matrix)  # Last matrix, for iterations[-1]
        return results
