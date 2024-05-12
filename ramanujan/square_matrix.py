from __future__ import annotations
from typing import Dict, List, Collection

from multimethod import multimethod

import sympy as sp


class SquareMatrix(sp.Matrix):
    """
    Represents an NxN marix.

    Inherits from sympy's matrix and supports all of its methods and operators:
    https://docs.sympy.org/latest/modules/matrices/matrices.html
    """

    def __repr__(self) -> str:
        return repr(sp.Matrix(self)).replace("Matrix", "SquareMatrix")

    def __str__(self) -> str:
        return repr(self)

    def __init__(self, *args, **kwargs):
        super().__init__()
        assert (
            self.rows == self.cols
        ), f"Only square NxN matrices are supported, received {self.rows}x{self.cols}"

    def __eq__(self, other: SquareMatrix) -> bool:
        return all(sp.simplify(cell) == 0 for cell in self - other)

    def __call__(self, *args, **kwargs) -> SquareMatrix:
        """
        Substitutes variables in the matrix, in a more math-like syntax.

        Calls the underlying sympy `subs` method:
        https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs
        """
        return self.subs(*args, **kwargs)

    def N(self) -> int:
        """
        Returns the amount of rows/columns of the square matrix (which is of dimension NxN)
        """
        return self.rows

    def gcd(self) -> sp.Rational:
        """
        Returns the rational gcd of the matrix, which could also be parameteric.
        """
        return sp.gcd(list(self))

    def normalize(self) -> SquareMatrix:
        """Normalizes the matrix by reducing its rational gcd"""
        m = self.simplify()
        return (m / m.gcd()).simplify()

    def inverse(self) -> Matrix:
        """
        Inverses the matrix.
        """
        return self.inv()

    def simplify(self) -> SquareMatrix:
        """Returns a simplified version of matrix"""
        return SquareMatrix(sp.simplify(self))

    @multimethod
    def walk(
        self, trajectory: Dict, iterations: Collection[int], start: Dict
    ) -> List[SquareMatrix]:
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
        """

        assert (
            start.keys() == trajectory.keys()
        ), "`start` and `trajectory` must contain same keys"

        iterations_set = set(iterations)
        assert len(iterations_set) == len(
            iterations
        ), "`iterations` values must be unique"

        position = start
        matrix = SquareMatrix.eye(2)
        results = []
        for i in range(
            max(iterations_set) + 1
        ):  # Plus one for the last requested `iterations` value
            if i in iterations:
                results.append(matrix)
            matrix *= self.subs(position)
            position = {key: trajectory[key] + value for key, value in position.items()}
        return results

    @multimethod
    def walk(
        self, trajectory: Dict, iterations: int, start: Dict
    ) -> SquareMatrix:  # noqa: F811
        return self.walk(trajectory, [iterations], start)[0]

    def as_pcf(self, deflate_all=True):
        """
        Converts a `SquareMatrix` to an equivalent `PCF`

        Args:
            deflate_all: if `True`, the function will also deflate the returned PCF to the fullest.
        Returns:
            a `PCFFRomSquareMatrix` object, containing a `PCF` whose limit is equal to
            a mobius transform of the original `SquareMatrix`.
        """
        from ramanujan.pcf import PCFFromSquareMatrix

        return PCFFromSquareMatrix(self, deflate_all)
