from __future__ import annotations

from typing import Dict, List, Union

from ramanujantools import Matrix
from ramanujantools.flint import FlintRational


class FlintMatrix:
    def __init__(
        self, rows: int, cols: int, values: List[FlintMatrix], symbols
    ) -> FlintMatrix:
        self.symbols = symbols
        self._rows = rows
        self._cols = cols
        self.values = values

    @staticmethod
    def from_sympy(matrix: Matrix, symbols=None) -> FlintMatrix:
        symbols = [str(symbol) for symbol in symbols or matrix.free_symbols]
        values = [FlintRational.from_sympy(cell, symbols) for cell in matrix]
        return FlintMatrix(matrix.rows, matrix.cols, values, symbols)

    @staticmethod
    def eye(N: int, symbols) -> FlintMatrix:
        values = [0] * N**2
        current = 0
        while current < N**2:
            values[current] = 1
            current += N + 1
        return FlintMatrix(N, N, values, symbols)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row = key[0]
            col = key[1]
            return self.values[row * self.cols() + col]
        else:
            return self.values[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row = key[0]
            col = key[1]
            self.values[row * self.cols() + col] = value
        else:
            self.values[key] = value

    def free_symbols(self):
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
        if isinstance(other, int):
            return self * other
        return other * self

    def __truediv__(self, other: int) -> FlintMatrix:
        if isinstance(other, FlintMatrix):
            raise ValueError("Attempted to divide by matrix!")
        return FlintMatrix(
            self.rows(),
            self.cols(),
            [value / other for value in self.values],
            self.free_symbols(),
        )

    def subs(self, substitutions: Dict) -> FlintMatrix:
        return FlintMatrix(
            self.rows(),
            self.cols(),
            [value.subs(substitutions) for value in self.values],
            self.free_symbols(),
        )
