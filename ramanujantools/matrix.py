from __future__ import annotations
from typing import Dict, List, Collection, Callable
from functools import lru_cache

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

    def __hash__(self) -> int:
        return hash(frozenset(self))

    def __call__(self, substitutions: Dict) -> Matrix:
        """
        Substitutes symbols in the matrix, in a more math-like syntax.
        """
        return self.subs(substitutions)

    def subs(self, substitutions: Dict) -> Matrix:
        """
        Substitutes symbols in the matrix.

        Calls the underlying sympy xreplace method, as subs is too slow and we don't need it's extra functionality
        https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.xreplace

        In the case where all substitutions are numbers (and not symbols or expressions),
        Uses numerical as an optimization
        """
        if self._can_call_numerical_subs(substitutions):
            return self.numerical_subs(substitutions)
        return self.xreplace(substitutions)

    def _can_call_numerical_subs(self, substitutions: Dict) -> bool:
        """
        Returns true iff the all substitutions are numerical and we can can call `numerical_subs` instead of `xreplace`.
        """
        return not (
            len(substitutions) < len(self.free_symbols)
            or any(isinstance(element, sp.Expr) for element in substitutions.values())
        )

    def numerical_subs(self, substitutions: Dict) -> Matrix:
        """
        An optimized version of `subs` for the case where all free_symbols are present in the substitutions dict,
        and all requested values are numerical (i.e not sympy expressions of any sort)
        """
        fast_subs = self.create_fast_subs(self)
        return fast_subs(substitutions)

    @staticmethod
    @lru_cache
    def create_fast_subs(matrix: Matrix) -> Callable:
        """
        Returns a function that evaluates the matrix at given substitutions.

        Works by storing the matrix as a string, setting all local symbols as variables in the local scope,
        and then simply (but not symply) returning it.
        Evaluation occures in the python interpreter, rather by recursively substituting the sympy expressions.
        This optimizes by a factor of 2~
        """
        matrix_string = str(matrix)

        def fast_subs(substitutions: Dict):
            for symbol, value in substitutions.items():
                exec(f"{symbol} = {value}")
            return eval(matrix_string)

        return fast_subs

    def is_square(self) -> int:
        """
        Returns the amount of rows/columns of the square matrix (which is of dimension NxN)
        """
        return self.rows == self.cols

    def denominator_lcm(self) -> sp.Expr:
        """
        Returns the lcm of all denominators
        """
        divisors = [cell.cancel().as_numer_denom()[1] for cell in self]
        return sp.lcm(divisors)

    def as_polynomial(self) -> Matrix:
        """
        Converts the matrix to a polynomial matrix by multiplying it by the denominators lcm.
        """
        return (self * self.denominator_lcm()).simplify()

    def gcd(self) -> sp.Rational:
        """
        Returns the rational gcd of the matrix, which could also be parameteric.
        """
        return sp.gcd(list(self))

    def reduce(self) -> Matrix:
        """
        Reduces gcd from the matrix
        """
        # important: must simplify first, both for correctness and performance. reproducible example:
        # t = x*(x - 1)*(x + 1)/(x**2 + x)
        # sp.gcd(t, x) == x
        # sp.gcd(t.simplify(), x) == 1
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

        if not all(depth >= 0 for depth in iterations):
            raise ValueError(
                f"iterations must contain only non-negative values, got {iterations}"
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
            matrix *= self(position)
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

        return PCFFromMatrix(self.as_polynomial(), deflate_all)
