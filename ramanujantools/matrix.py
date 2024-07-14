from __future__ import annotations
from typing import Dict, List, Collection

from multimethod import multimethod

import sympy as sp
from sympy.abc import n


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

    def coboundary(self, U: Matrix, symbol: sp.Symbol = n) -> Matrix:
        r"""
        Calculates the coboundary relation of M and U $U(n) * M(n) * U^{-1}(n+1)$, where $M$ is `self`.

        Args:
            U: The coboundary matrix
            symbol: The symbol to use when calculating the coboundary relation. `n` by default.
        Returns:
            The coboundary relation as described above
        """
        return (U * self * U.inverse().subs({symbol: symbol + 1})).simplify()

    def companion_coboundary_matrix(self, symbol: sp.Symbol = n) -> Matrix:
        r"""
        Constructs a new matrix U such that `self.coboundary(U)` is a companion matrix.
        """
        if not (self.is_square()):
            raise ValueError("Only square matrices can have a coboundary relation")
        N = self.rows
        e1 = Matrix.zeros(N, 1)
        e1[0, 0] = 1
        vectors = [e1]
        for i in range(1, N):
            vectors.append(self * vectors[i - 1].subs({symbol: symbol + 1}))
        return Matrix.hstack(*vectors).inverse().simplify()

    def is_companion(self) -> bool:
        r"""
        Returns True iff the matrix is a companion matrix.
        """
        if not self.is_square():
            raise ValueError("Attempted to check if a non-square matrix is companion")
        N = self.rows
        for row in range(N):
            for col in range(N - 1):
                if row == col + 1:
                    if self[row, col] != 1:
                        return False
                else:
                    if self[row, col] != 0:
                        return False
        return True

    @staticmethod
    def inflation_coboundary_matrix(
        N: int, c: sp.Expr, symbol: sp.Symbol = n
    ) -> Matrix:
        r"""
        Returns the matrix inflation matrix U for polynomial c.

        See `inflate`.

        Args:
            N: The dimension of the square matrix.
            c: The polynomial to inflate by.
            symbol: The symbol of the coboundary relation.

        Returns:
            The inflation matrix U
        """
        U = Matrix.eye(N)
        for i in range(1, N):
            U[N - (i + 1), N - (i + 1)] = U[N - i, N - i] * c.subs({symbol: symbol - i})
        return U

    def inflate(self, c: sp.Expr, symbol: sp.Symbol = n) -> Matrix:
        r"""
        Inflates the matrix by polynomial c.

        Inflated matrix $M'(n)$ satisfies $M'(n) = c(n) * U(n) * M(n) * U^{-1}(n+1)$

        Args:
            c: The polynomial to inflate by.
            symbol: The symbol of the coboundary relation (n in the example).
        Returns:
            The inflated matrix as defined above.
        Raises:
            ValueError: if the matrix is not a square matrix.
        """
        if not self.is_square():
            raise ValueError("Can only inflate square matrices")
        return c * self.coboundary(
            Matrix.inflation_coboundary_matrix(N=self.rows, c=c, symbol=symbol)
        )

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

        return PCFFromMatrix(self.as_polynomial(), deflate_all)
