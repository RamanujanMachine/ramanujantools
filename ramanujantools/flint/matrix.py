from __future__ import annotations

from typing import Dict, List, Union

import sympy as sp
from multimethod import multimethod

from ramanujantools import Matrix, Position
from ramanujantools.flint import FlintRational


class FlintMatrix:
    """
    Represents a Matrix of FlintRationals.

    It's logic is limited compared to the main Matrix, as it's designed for bottlenecks.
    """

    def __init__(
        self, rows: int, cols: int, values: List[FlintMatrix], symbols
    ) -> FlintMatrix:
        self.symbols = symbols
        self._rows = rows
        self._cols = cols
        self.values = values

    @staticmethod
    def from_sympy(matrix: Matrix, symbols=None) -> FlintMatrix:
        """
        Converts a Matrix to FlintMatrix.
        """
        symbols = [str(symbol) for symbol in symbols or matrix.free_symbols]
        values = [FlintRational.from_sympy(cell, symbols) for cell in matrix]
        return FlintMatrix(matrix.rows, matrix.cols, values, symbols)

    @staticmethod
    def eye(N: int, symbols) -> FlintMatrix:
        """
        Creates an identity matrix of size N.

        Args:
            N: The squared matrix dimension
            symbols: The symbols this matrix supports
        """
        values = [FlintRational.from_sympy(sp.simplify(0), symbols)] * N**2
        current = 0
        while current < N**2:
            values[current] += 1
            current += N + 1
        return FlintMatrix(N, N, values, symbols)

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

    def free_symbols(self):
        """
        Returns the free symbols this matrix supports
        """
        return self.symbols

    def rows(self):
        return self._rows

    def cols(self):
        return self._cols

    def shape(self):
        return (self.rows(), self.cols())

    def row(self, index: int) -> List[FlintRational]:
        row = []
        for i in range(self.cols()):
            row.append(self[index, i])
        return row

    def col(self, index: int) -> List[FlintRational]:
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
        return f"FlintMatrix({self.data()})"

    def __str__(self) -> str:
        return f"FlintMatrix({self.data()})"

    def __mul__(self, other: Union[FlintMatrix, int]) -> FlintMatrix:
        """
        Multiplies self by another FlintMatrix or a scalar.
        """
        if isinstance(other, FlintMatrix):
            if self.cols() != other.rows():
                raise ValueError("Attempting to multiply")
            elements = []
            for row in range(self.rows()):
                for col in range(other.cols()):
                    current = 0
                    for k in range(self.cols()):
                        current += self[row, k] * other[k, col]
                    elements.append(current)
            return FlintMatrix(self.rows(), self.cols(), elements, self.free_symbols())

        else:
            return FlintMatrix(
                self.rows(),
                self.cols(),
                [value * other for value in self.values],
                self.free_symbols(),
            )

    def __rmul__(self, other: Union[FlintMatrix, int]) -> FlintMatrix:
        """
        Multiplies a FlintMatrix or a scalar by self.
        """
        if isinstance(other, int):
            return self * other
        return other * self

    def __truediv__(self, other: int) -> FlintMatrix:
        """
        Divides self by a scalar
        """
        if isinstance(other, FlintMatrix):
            raise ValueError("Attempted to divide by matrix!")
        return FlintMatrix(
            self.rows(),
            self.cols(),
            [value / other for value in self.values],
            self.free_symbols(),
        )

    def subs(self, substitutions: Dict) -> FlintMatrix:
        """
        Substitutes symbols in the matrix.
        """
        return FlintMatrix(
            self.rows(),
            self.cols(),
            [value.subs(substitutions) for value in self.values],
            self.free_symbols(),
        )

    def factor(self) -> Matrix:
        """
        Factors all elements in the matrix.
        """
        values = [value.factor() for value in self.values]
        return Matrix(self.rows(), self.cols(), values)

    @multimethod
    def walk(self, trajectory: Dict, iterations: List[int], start: Dict) -> FlintMatrix:
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
        matrix = FlintMatrix.eye(self.rows(), self.free_symbols())
        for depth in range(0, iterations[-1]):
            if depth in iterations:
                results.append(matrix)
            matrix *= self.subs(position)
            position += trajectory
        results.append(matrix)  # Last matrix, for iterations[-1]
        return results

    @multimethod
    def walk(  # noqa: F811
        self,
        trajectory: Dict,
        iterations: int,
        start: Dict,
    ) -> Matrix:
        return self.walk(trajectory, [iterations], start)[0]
