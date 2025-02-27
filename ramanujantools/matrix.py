from __future__ import annotations
from typing import Dict, List, Set, Callable, Tuple
from functools import lru_cache, cached_property

from multimethod import multimethod

import numpy as np
import mpmath as mp
import sympy as sp
from sympy.abc import n

from ramanujantools import Position
from ramanujantools.flint_core import mpoly_ctx, FlintMatrix


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

    def _can_call_flint_walk(self, trajectory: Dict, start: Dict) -> bool:
        trajectory = Position(trajectory)
        start = Position(start)

        subbed_in = trajectory.free_symbols().union(start.free_symbols())
        subbed_out = set(trajectory.keys()).union(set(start.keys()))

        return (self.free_symbols - subbed_out).union(subbed_in) != set()

    def _can_call_numerical_subs(self, substitutions: Dict) -> bool:
        """
        Returns true iff the all substitutions are numerical and we can can call `numerical_subs` instead of `xreplace`.
        """
        substitutions = Position(substitutions)
        return (
            substitutions.keys() == self.free_symbols
            and self.is_polynomial()
            and substitutions.is_integer()
        )

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

    @lru_cache
    def inverse(self) -> Matrix:
        """
        Inverts the matrix.
        """
        return self.inv()

    @lru_cache
    def simplify(self) -> Matrix:
        """
        Returns a simplified version of matrix
        """
        return Matrix(sp.simplify(self))

    def factor(self) -> Matrix:
        return FlintMatrix.from_sympy(
            self, mpoly_ctx(self.free_symbols, fmpz=True)
        ).factor()

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
        Calculates the coboundary relation of M and U $U^{-1}(n) * M(n) * U^(n+1)$, where $M$ is `self`.

        Args:
            U: The coboundary matrix
            symbol: The symbol to use when calculating the coboundary relation. `n` by default.
        Returns:
            The coboundary relation as described above
        """
        return (U.inverse() * self * U.subs({symbol: symbol + 1})).simplify()

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
        return Matrix.hstack(*vectors).simplify()

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

    def as_companion(self) -> Matrix:
        r"""
        Converts the matrix to companion form.

        Args:
            inflate_all: if True, will greedily inflate the companion form matrix until it's polynomial.
        """
        U = self.companion_coboundary_matrix()
        rank = U.rank()
        if rank == self.rows:
            return self.coboundary(self.companion_coboundary_matrix())
        else:
            symbols = sp.symbols(f"c:{rank}")
            variables = Matrix(symbols + (-1,))
            truncated = U[:, : rank + 1]
            solutions = sp.solve(truncated * variables, symbols)
            elements = [solutions[symbol] for symbol in symbols]
            return Matrix.companion_form(elements)

    @lru_cache
    def _walk_inner(
        self,
        trajectory: Position,
        iterations: Tuple[int],
        start: Position,
    ) -> List[Matrix]:
        """
        Internal walk function, used for type conversions and for caching. Do not use directly.
        """
        from ramanujantools.flint_core import FlintMatrix

        if self._can_call_flint_walk(trajectory, start):
            symbols = self.walk_free_symbols(start)
            as_flint = FlintMatrix.from_sympy(
                self, mpoly_ctx(symbols, fmpz=start.is_polynomial())
            )
            results = as_flint.walk(trajectory, list(iterations), start)
            results = [result.factor() for result in results]
            return results
        else:
            results = []
            position = start.copy()
            matrix = Matrix.eye(self.rows)
            for depth in range(0, iterations[-1]):
                if depth in iterations:
                    results.append(matrix)
                matrix *= self(position)
                position += trajectory
            results.append(matrix)  # Last matrix, for iterations[-1]
            return results

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
        return self._walk_inner(
            Position(trajectory), tuple(iterations), Position(start)
        )

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

    @staticmethod
    def poincare_poly(poly: sp.PurePoly) -> sp.PurePoly:
        """
        Deflates a polynomial such that all coefficients approach a finite number.
        Assumes polynomial only contain n as a free symbol.
        """
        current_degree = 0
        charpoly_coeffs = poly.all_coeffs()
        for i in range(len(charpoly_coeffs)):
            coeff = charpoly_coeffs[i]
            numerator, denominator = coeff.as_numer_denom()
            degree = sp.Poly(numerator, n).degree() - sp.Poly(denominator, n).degree()
            if (current_degree * i) < degree:
                current_degree = -(degree // -i)  # ceil div trick
        coeffs = [
            (charpoly_coeffs[i] / (n ** (current_degree * i))).limit(n, "oo")
            for i in range(len(charpoly_coeffs))
        ]
        return sp.PurePoly(coeffs, poly.gen)

    def charpoly(self, poincare=False, *args, **kwargs) -> sp.PurePoly:
        """
        Returns the characteristic polynomial of the matrix.

        Calls the underyling sympy.Matrix.charpoly and supports its arguments:
        https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrixbase.MatrixBase.charpoly

        Args:
            poincare: if True, converts the polynomial to Poincare form. False by default.
        """
        poly = super().charpoly(*args, **kwargs)
        if poincare:
            poly = Matrix.poincare_poly(poly)
        return poly

    def eigenvals(self, poincare=False) -> Dict:
        """
        Returns the eigenvalues of the matrix, which are the roots of the characteristic polynomials.

        Args:
            poincare: if True, returns the eigenvalues of the Poincare characteristic polynomial instead.
            False by default.
        """
        charpoly = self.charpoly(poincare)
        return sp.roots(charpoly)

    def sorted_eigenvals(self) -> List:
        """
        Returns the eigenvalues of the matrix in Poincare form, sorted by absolute value in descending order.
        """
        unsorted = self.eigenvals(poincare=True)
        retval = []
        for key, value in unsorted.items():
            retval += [key] * value
        return sorted(
            retval, key=lambda value: abs(value).evalf(chop=True), reverse=True
        )

    def errors(self) -> List:
        """
        Approximate the possible errors of integer recurrence approximations using this Matrix.
        """
        lambdas = [e.evalf(chop=True) for e in self.sorted_eigenvals()]
        deltas = []
        for i in range(1, len(lambdas)):
            deltas.append(sp.log(abs(lambdas[0]) / abs(lambdas[i])))
        return deltas

    def gcd_slope(self, depth=20) -> mp.mpf:
        r"""
        Attempts to perform a linear fit of $\bar{q} = \frac{q_n}{gcd(p_n, q_n)}$ as a function of $n$.

        Args:
            depth: The maximal value of $n$
        Returns:
            The slope of the lienar fit of $\bar{q}$.
        """
        depths = list(range(1, depth))
        q_reduced_list = []
        limits = self.limit({n: 1}, depths, {n: 1})
        for limit in limits:
            p, q = limit.as_rational()
            gcd = sp.gcd(p, q)
            q_reduced_list.append(sp.log(abs(q // gcd)).evalf(30))
        fit = np.polyfit(
            np.array(depths), np.array(q_reduced_list, dtype=np.float64), 1
        )
        return mp.mpf(fit[0])

    def kamidelta(self, depth=20):
        r"""
        Predicts the possible delta values of the integer sequence approximation that the matrix generates.

        The irrationality measure $\delta$ is defined as $|\frac{p_n}{q_n} - L| = \frac{1}{\bar{q_n}^{1+\delta}}$.
        As one can tell, the delta is determined by two values:
        The approximation error $L-\frac{p_n}{q_n}$,
        and the gcd of both sequences $gcd(p_n,q_n)$.

        This algorithm approximates both values in order to predict possible delta values.

        Args:
            depth: The maximal depth for the gcd slope fit.
        Returns:
            A list (of size N - 1) containing all predicted delta possibilities.
        """
        errors = self.errors()
        slope = self.gcd_slope(depth)
        return [-1 + error / slope for error in errors]
