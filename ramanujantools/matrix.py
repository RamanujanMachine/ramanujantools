from __future__ import annotations
from typing import Dict, List, Set, Callable, Union
from functools import lru_cache, cached_property

from multimethod import multimethod

import sympy as sp
from sympy.abc import n

from ramanujantools import Position


class Matrix(sp.Matrix):
    """
    Represents a Matrix.

    Inherits from sympy's matrix and supports all of its methods and operators:
    https://docs.sympy.org/latest/modules/matrices/matrices.html
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @staticmethod
    def e(N: int, index: int, column=True) -> Matrix:
        r"""
        Returns a coordinate vector of size N for a given index, i.e,
        a vector of size N of zeroes with 1 in the corresponding index

        Args:
            N: The vector size
            index: The index of the given axis
            column: will return the vector in column form if true, in row form otherwise.
        Returns:
            The desired coordinate vector described above
        """
        if index >= N:
            raise ValueError(f"Cannot create {index}th axis vector of size {N}")
        if column:
            return Matrix.eye(N).col(index)
        else:
            return Matrix.eye(N).row(index)

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Matrix) -> bool:
        return (
            self.rows == other.rows
            and self.cols == other.cols
            and all(sp.simplify(cell) == 0 for cell in self - other)
        )

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

    def _is_numeric(self, substitutions: Dict) -> bool:
        """
        Returns true iff all substitutions are numerical
        """
        return len(substitutions) == len(self.free_symbols) and not (
            any(isinstance(element, sp.Expr) for element in substitutions.values())
        )

    def _can_call_numerical_subs(self, substitutions: Dict) -> bool:
        """
        Returns true iff the all substitutions are numerical and we can can call `numerical_subs` instead of `xreplace`.
        """
        return self.is_polynomial() and self._is_numeric(substitutions)

    def numerical_subs(self, substitutions: Dict) -> Matrix:
        """
        An optimized version of `subs` for the case where all free_symbols are present in the substitutions dict,
        and all requested values are numerical (i.e not sympy expressions of any sort)
        """
        fast_subs = Matrix.create_fast_subs(self)
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

    @cached_property
    def denominator_lcm(self) -> sp.Expr:
        """
        Returns the lcm of all denominators
        """
        divisors = [cell.cancel().as_numer_denom()[1] for cell in self]
        return sp.lcm(divisors)

    def is_polynomial(self) -> bool:
        return self.denominator_lcm == 1

    def as_polynomial(self) -> Matrix:
        """
        Converts the matrix to a polynomial matrix by multiplying it by the denominators lcm.
        """
        return (self * self.denominator_lcm).simplify()

    @cached_property
    def gcd(self) -> sp.Rational:
        """
        Returns the rational gcd of the matrix, which could also be parameteric.
        """
        return sp.gcd(list(self.simplify()))

    def reduce(self) -> Matrix:
        """
        Reduces gcd from the matrix
        """
        # important: must simplify first, both for correctness and performance. reproducible example:
        # t = x*(x - 1)*(x + 1)/(x**2 + x)
        # sp.gcd(t, x) == x
        # sp.gcd(t.simplify(), x) == 1
        return (self / self.gcd).simplify()

    def limit_equivalent(self, other: Matrix) -> bool:
        """
        Returns true iff two matrices are limit equivalent.
        Two matrices are limit equivalent iff self = c * other for some c.
        """
        return self.as_polynomial().reduce() == other.as_polynomial().reduce()

    @lru_cache(maxsize=32)
    def inverse(self) -> Matrix:
        """
        Inverts the matrix.
        """
        return self.inv()

    @lru_cache(maxsize=32)
    def simplify(self) -> Matrix:
        """
        Returns a simplified version of matrix
        """
        return Matrix(sp.simplify(self))

    def factor(self) -> Matrix:
        from ramanujantools.flint import FlintMatrix

        return FlintMatrix.from_sympy(self, self.free_symbols).factor()

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

    @staticmethod
    def companion_form(values: List[sp.Expr]) -> Matrix:
        N = len(values)
        columns = []
        for i in range(N - 1):
            columns.append(Matrix.e(N, i + 1))
        columns.append(Matrix(list(values)))
        return Matrix.hstack(*columns)

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

        Inflated matrix $M'(n)$ satisfies $M'(n) = c(n) * U(n) * M(n) * U^{-1}(n+1)$,
        Where `U` is the inflation matrix.
        Inflation is defined by the case where M is a companion matrix,

        $M(n) =
        \begin{bmatrix}
            0&0&\cdots & 0 & 0 & p_{N}(n) \cr
            1&0&\cdots & 0 & 0 & p_{N-1}(n) \cr
            0&1&\cdots & 0 & 0 & p_{N-2}(n) \cr
            \vdots & &\ddots  &  & \vdots & \vdots \cr
            0&0&\cdots & 1 & 0 & p_{2}(n) \cr
            0&0&\cdots & 0 & 1 & p_{1}(n) \cr
        \end{bmatrix}$

        Then the inflated matrix satisfies

        $M'(n) =
        \begin{bmatrix}
            0&0&\cdots & 0 & 0 & p_{N}(n)\prod_{i=0}^{N-1}c(n-i) \cr
            1&0&\cdots & 0 & 0 & p_{N-1}(n)\prod_{i=0}^{N-2}c(n-i) \cr
            0&1&\cdots & 0 & 0 & p_{N-2}(n)\prod_{i=0}^{N-3}c(n-i) \cr
            \vdots & &\ddots  &  & \vdots & \vdots \cr
            0&0&\cdots & 1 & 0 & p_{2}(n)c(n)c(n-1) \cr
            0&0&\cdots & 0 & 1 & p_{1}(n)c(n) \cr
        \end{bmatrix}$

        See `inflation_coboundary_matrix`

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
        c = sp.simplify(c)
        m = c * self.coboundary(
            Matrix.inflation_coboundary_matrix(N=self.rows, c=c, symbol=symbol)
        )
        return m.simplify()

    def deflate(self, c: sp.Expr, symbol: sp.Symbol = n) -> Matrix:
        r"""
        Deflates the matrix by polynomial c.

        This is a syntactic sugar to `self.inflate(1/c, symbol)`.
        See `inflate`.
        """

        return self.inflate(c=1 / c, symbol=symbol)

    def canonize_companion(self) -> Matrix:
        r"""
        canonizes the companion matrix to a canonical form.

        The canonical form is achieved when $p_{1}(n) = 1$ using inflations and deflations.

        Returns:
            The matrix in normalized companion form.
        Raises:
            ValueError: If `self` is not a companion matrix.
        """
        if not (self.is_companion()):
            raise ValueError(
                "Companion normalization can be used only on companion matrices"
            )
        return self.deflate(self[-1, -1])

    def companion_equivalent(self, other: Matrix) -> bool:
        r"""
        Returns true iff both companion matrices are equivalent up to inflations.

        Returns:
            True iff they are equivalent
        Raises:
            ValueError: if one of the matrices are not a companion matrix
        """
        if not (self.is_companion()):
            raise ValueError(
                f"Attempted to compare companion matrices but received non-companion self={self}"
            )

        if not (other.is_companion()):
            raise ValueError(
                f"Attempted to compare companion matrices but received non-companion other={other}"
            )

        return self.canonize_companion() == other.canonize_companion()

    @staticmethod
    def select_inflation_factor(factors: Set[sp.Expr], polynomial_index: int):
        r"""
        Returns the next factor out of `factors` to inflate during companion normalization.

        Implemented as a greedy algorithm: in the ith row, given a set of factors to inflate,
        for every factor f in `factors` we check if any of [f+1, ..., f+(i-1)] needs factoring.
        If that's the case, we shouldn't inflate by f, as it will be inflated anyway.
        Otherwise, choose f for the next inflation.

        Future improvement:
        Suppose we want to inflate by f. We can select any of [f, f+1, ..., f+(i-1)].
        A nice optimization would be to look at deeper rows and see which one is optimal.
        """
        # sorting lexicographically for deterministic behavior
        for factor in sorted(list(factors), key=str):
            inflation_candidates = [factor + j for j in range(1, polynomial_index + 1)]
            # check if there exists a `factor + j` such that it will also inflate `factor`
            if all(candidate not in factors for candidate in inflation_candidates):
                return factor

    def normalize_companion(self) -> Matrix:
        r"""
        Inflates a (rational) companion matrix by a minimal polynomial to make it polynomial
        """

        def get_next_denominator(m: Matrix):
            for row in reversed(range(m.rows)):
                denominator = m[row, -1].cancel().as_numer_denom()[1].expand()
                if denominator != 1:
                    return m.rows - row, denominator
            return 0, 1

        m = self.canonize_companion()
        original = m
        row, denominator = get_next_denominator(m)
        while denominator != 1:
            content, factors = sp.factor_list(denominator)
            factors = set(dict(factors))
            m = m.inflate(content * Matrix.select_inflation_factor(factors, row))
            row, denominator = get_next_denominator(m)
            assert m.companion_equivalent(original)
        return m.simplify()

    def as_companion(self, inflate_all=True) -> Matrix:
        r"""
        Converts the matrix to companion form.

        Args:
            inflate_all: if True, will greedily inflate the companion form matrix until it's polynomial.
        """
        companion = self.coboundary(self.companion_coboundary_matrix())
        if inflate_all:
            companion = companion.normalize_companion()
        return companion

    @multimethod
    def walk(  # noqa: F811
        self,
        trajectory: Dict,
        iterations: List[int],
        start: Dict,
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
        from ramanujantools.flint import FlintMatrix

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

        if not iterations == sorted(iterations):
            raise ValueError(f"Iterations must be sorted, got {iterations}")

        if not all(depth >= 0 for depth in iterations):
            raise ValueError(
                f"iterations must contain only non-negative values, got {iterations}"
            )

        position = Position(start)
        trajectory = Position(trajectory)

        if self._is_numeric(position + trajectory):
            results = []
            matrix = Matrix.eye(self.rows)
            for depth in range(0, iterations[-1]):
                if depth in iterations:
                    results.append(matrix)
                matrix *= self(position)
                position += trajectory
            results.append(matrix)  # Last matrix, for iterations[-1]
            return results
        else:
            symbols = self.walk_free_symbols(start)
            as_flint = FlintMatrix.from_sympy(self, symbols)
            results = as_flint.walk(trajectory, iterations, start)
            return [result.factor() for result in results]

    @multimethod
    def walk(  # noqa: F811
        self,
        trajectory: Dict,
        iterations: int,
        start: Dict,
    ) -> Matrix:
        return self.walk(trajectory, [iterations], start)[0]

    def walk_free_symbols(self, start: Dict) -> Set:
        """
        Returns the expected free_symbols of the expression `self.walk(trajectory, iterations, start)`
        """

        free_symbols = self.free_symbols.copy()
        for key in start.keys():
            free_symbols = free_symbols.union(set(sp.simplify(key).free_symbols))
        for value in start.values():
            free_symbols = free_symbols.union(set(sp.simplify(value).free_symbols))
        return free_symbols

    @multimethod
    def limit(
        self,
        trajectory: Dict,
        iterations: List[int],
        start: Dict,
    ):  # noqa: F811
        from ramanujantools import Limit

        def walk_function(iterations):
            return self.walk(trajectory, iterations, start)

        return Limit.walk_to_limit(iterations, walk_function)

    @multimethod
    def limit(  # noqa: F811
        self,
        trajectory: Dict,
        iterations: int,
        start: Dict,
    ):
        return self.limit(trajectory, [iterations], start)[0]

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
