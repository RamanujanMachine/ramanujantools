from __future__ import annotations

from typing import Dict, List, Collection
from multimethod import multimethod

import sympy as sp


class Matrix(sp.Matrix):
    """
    Represens a marix.

    Inherits from sympy's matrix and supports all of its methods and operators:
    https://docs.sympy.org/latest/modules/matrices/matrices.html
    """

    def __eq__(self, other: Matrix) -> bool:
        return all(sp.simplify(cell) == 0 for cell in self - other)

    def __call__(self, *args, **kwargs) -> Matrix:
        """
        Substitutes variables in the matrix, in a more math-like syntax.

        Calls the underlying sympy `subs` method:
        https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs
        """
        return self.subs(*args, **kwargs)

    def gcd(self) -> sp.Rational:
        """
        Returns the rational gcd of the matrix
        """
        primitives = [
            primitive for primitive, _ in list(map(lambda cell: cell.primitive(), self))
        ]
        return sp.gcd(primitives)

    def normalize(self) -> Matrix:
        """Normalizes the matrix by reducing its rational gcd"""
        gcd = self.gcd()
        return self / gcd

    def simplify(self) -> Matrix:
        """Returns a simplified version of matrix"""
        return Matrix(sp.simplify(self))

    @multimethod
    def walk(
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
        """

        assert (
            start.keys() == trajectory.keys()
        ), "`start` and `trajectory` must contain same keys"

        iterations_set = set(iterations)
        assert len(iterations_set) == len(
            iterations
        ), "`iterations` values must be unique"

        position = start
        matrix = Matrix.eye(2)
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
    ) -> Matrix:  # noqa: F811
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


def zero():
    r"""Returns the zero vector $\begin{pmatrix} 0 \cr 1 \end{pmatrix}$"""
    return Matrix([0, 1])


def inf():
    r"""Returns the infinity vector $\begin{pmatrix} 1 \cr 0 \end{pmatrix}$"""
    return Matrix([1, 0])
