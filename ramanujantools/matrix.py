from __future__ import annotations
from typing import Dict, List, Collection

from multimethod import multimethod

import sympy as sp


class Matrix(sp.Matrix):
    """
    Represents a Matrix.

    Inherits from sympy's matrix and supports all of its methods and operators:
    https://docs.sympy.org/latest/modules/matrices/matrices.html
    """

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Matrix) -> bool:
        return all(sp.simplify(cell) == 0 for cell in self - other)

    def __call__(self, *args, **kwargs) -> Matrix:
        """
        Substitutes variables in the matrix, in a more math-like syntax.

        Calls the underlying sympy `subs` method:
        https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs
        """
        return self.subs(*args, **kwargs)

    def is_square(self) -> int:
        """
        Returns the amount of rows/columns of the square matrix (which is of dimension NxN)
        """
        return self.rows == self.cols

    def gcd(self) -> sp.Rational:
        """
        Returns the rational gcd of the matrix, which could also be parameteric.
        """
        return sp.gcd(list(self))

    def normalize(self) -> Matrix:
        """
        Normalizes the matrix by reducing its rational gcd
        """
        m = self.simplify()
        return (m / m.gcd()).simplify()

    def inverse(self) -> Matrix:
        """
        Inverses the matrix.
        """
        return self.inv()

    def simplify(self) -> Matrix:
        """
        Returns a simplified version of matrix
        """
        return Matrix(sp.simplify(self))

    def singular_points(self) -> List[Dict]:
        r"""
        Calculates the singular points of the matrix,
        i.e, points where $|m| = 0$

        Returns:
            A list of substitution dicts that result in the matrix having a zero determinant.
            That is, for each dict in result, `self.subs(dict).det() == 0`
        """
        return sp.solve(self.det(), dict=True)

    @multimethod
    def walk(  # noqa: F811
        self, trajectory: Dict, iterations: Collection[int], start: Dict
    ) -> List[Matrix]:
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
        if not self.is_square():
            raise ValueError(
                f"Matrix.walk is only supported for square matrices, got a {self.rows}x{self.cols} matrix"
            )

        if start.keys() != trajectory.keys():
            raise ValueError(
                f"`start` and `trajectory` must contain same keys, got "
                f"start={set(start.keys())}, trajectory={set(trajectory.keys())}"
            )

        iterations_set = set(iterations)
        if len(iterations_set) != len(iterations):
            raise ValueError(f"`iterations` values must be unique, got {iterations}")

        position = start
        matrix = Matrix.eye(self.rows)
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
    def walk(  # noqa: F811
        self, trajectory: Dict, iterations: int, start: Dict
    ) -> Matrix:
        return self.walk(trajectory, [iterations], start)[0]

    def as_pcf(self, deflate_all=True):
        """
        Converts a `Matrix` to an equivalent `PCF`

        Args:
            deflate_all: if `True`, the function will also deflate the returned PCF to the fullest.
        Returns:
            a `PCFFRomMatrix` object, containing a `PCF` whose limit is equal to
            a mobius transform of the original `Matrix`.
        """
        from ramanujantools.pcf import PCFFromMatrix

        return PCFFromMatrix(self, deflate_all)
