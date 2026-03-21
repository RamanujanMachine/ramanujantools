from __future__ import annotations

from typing import TYPE_CHECKING
from functools import lru_cache, cached_property

import numpy as np
import mpmath as mp
import sympy as sp
from sympy.abc import n

from ramanujantools import Position
from ramanujantools.utils import batched, Batchable
from ramanujantools.flint_core import flint_ctx, SymbolicMatrix, NumericMatrix

if TYPE_CHECKING:
    from ramanujantools import Limit
    from ramanujantools.asymptotics import GrowthRate, Reducer, SeriesMatrix


class Matrix(sp.Matrix):
    """
    Represents a Matrix.

    Inherits from sympy's matrix and supports all of its methods and operators:
    https://docs.sympy.org/latest/modules/matrices/matrices.html
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @staticmethod
    def e(N: int, index: int) -> Matrix:
        """
        Returns a coordinate vector of size N for a given index, i.e,
        a vector of size N of zeroes with 1 in the corresponding index

        Args:
            N: The vector size
            index: The index of the given axis
        Returns:
            The desired coordinate vector described above
        """
        if index >= N:
            raise ValueError(f"Cannot create {index}th axis vector of size {N}")
        return Matrix.eye(N).col(index)

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def _optimized_eq(cls, a, b) -> bool:
        return (
            a.rows == b.rows
            and a.cols == b.cols
            and all(cell == 0 for cell in (a - b).factor())
        )

    def __eq__(self, other: Matrix) -> bool:
        try:
            return self._optimized_eq(self, other)
        except sp.PolynomialError:
            return sp.Matrix.__eq__(self, other)

    def __hash__(self) -> int:
        return hash(frozenset(self))

    def __call__(self, substitutions: dict) -> Matrix:
        """
        Substitutes symbols in the matrix, in a more math-like syntax.
        """
        return self.subs(substitutions)

    def subs(self, substitutions: Position) -> Matrix:
        """
        Substitutes symbols in the matrix.

        Calls the underlying sympy xreplace method, as subs is too slow and we don't need it's extra functionality
        https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.xreplace
        """
        return self.xreplace(substitutions)

    def _is_numeric_walk(self, trajectory: dict, start: dict) -> bool:
        trajectory = Position(trajectory)
        start = Position(start)

        subbed_in = trajectory.free_symbols().union(start.free_symbols())
        subbed_out = set(trajectory.keys()).union(set(start.keys()))

        return (self.free_symbols - subbed_out).union(subbed_in) == set()

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

    def equal_projectively(self, other: Matrix) -> bool:
        """
        Returns true iff two matrices are equal projectively.
        Two matrices are equal projectively iff self = c * other for some c.
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
        if self.free_symbols:
            return SymbolicMatrix.from_sympy(
                self, flint_ctx(self.free_symbols, fmpz=True)
            ).factor()
        else:
            return self

    def singular_points(self) -> list[dict]:
        r"""
        Calculates the singular points of the matrix,
        i.e, points where $|m| = 0$

        Returns:
            A list of substitution dicts t+hat result in the matrix having a zero determinant.
            That is, for each dict in result, `self.subs(dict).det() == 0`
        """
        return sp.solve(self.det(), dict=True)

    def coboundary(self, U: Matrix, symbol: sp.Symbol = n, sign: bool = True) -> Matrix:
        r"""
        Calculates the coboundary relation of M and U $U^{-1}(n) * M(n) * U^(n+s)$,
        where $M$ is `self` and $s$ is `sign`.

        Args:
            U: The coboundary matrix
            symbol: The symbol to use when calculating the coboundary relation. `n` by default.
            sign: Represents if `symbol` goes up or down in the coboundary relation.
        Returns:
            The coboundary relation as described above
        """
        free_symbols = self.free_symbols.union(U.free_symbols).union({symbol})
        ctx = flint_ctx(free_symbols, fmpz=True)
        return (
            SymbolicMatrix.from_sympy(U.inverse(), ctx)
            * SymbolicMatrix.from_sympy(self, ctx)
            * SymbolicMatrix.from_sympy(
                U.subs({symbol: symbol + (1 if sign else -1)}), ctx
            )
        ).factor()

    @lru_cache
    def companion_coboundary_matrix(self, symbol: sp.Symbol = n) -> Matrix:
        r"""
        Constructs a new matrix U such that `self.coboundary(U)` is a companion matrix.
        """
        if not (self.is_square()):
            raise ValueError("Only square matrices can have a coboundary relation")
        N = self.rows
        free_symbols = self.free_symbols.union({symbol})
        ctx = flint_ctx(free_symbols, fmpz=True)
        flint_self = SymbolicMatrix.from_sympy(self, ctx)
        vectors = [SymbolicMatrix.from_sympy(Matrix(N, 1, [1] + (N - 1) * [0]), ctx)]
        for _ in range(1, N):
            vectors.append(flint_self * vectors[-1].subs({symbol: symbol + 1}))
        return Matrix.hstack(*[vector.factor() for vector in vectors]).factor()

    @staticmethod
    def companion_form(values: list[sp.Expr]) -> Matrix:
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

    def as_companion(self, symbol: sp.Symbol = n) -> Matrix:
        r"""
        Converts the matrix to companion form.
        """
        if symbol not in self.free_symbols:
            raise ValueError(
                f"Companionization symbol must be in matrix! matrix={self}, symbol={symbol}"
            )
        U = self.companion_coboundary_matrix(symbol)
        try:
            return self.coboundary(U, symbol)
        except ValueError:
            rank = U.rank()
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
        iterations: tuple[int],
        start: Position,
    ) -> list[Matrix]:
        """
        Internal walk function, used for type conversions and for caching. Do not use directly.
        """
        iterations = list(iterations)
        if self._is_numeric_walk(trajectory, start):
            results = NumericMatrix.walk(self, trajectory, iterations, start)
            return [result.to_rt() for result in results]
        else:
            symbols = self.walk_free_symbols(start)
            as_flint = SymbolicMatrix.from_sympy(
                self, flint_ctx(symbols, fmpz=start.is_polynomial())
            )
            results = as_flint.walk(trajectory, iterations, start)
            return [result.factor() for result in results]

    @batched("iterations")
    def walk(
        self,
        trajectory: dict,
        iterations: Batchable[int],
        start: dict,
    ) -> Batchable[Matrix]:
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

    def walk_free_symbols(self, start: dict) -> set[sp.Symbol]:
        """
        Returns the expected free_symbols of the expression `self.walk(trajectory, iterations, start)`
        """
        free_symbols = self.free_symbols.copy()
        for key in start.keys():
            free_symbols = free_symbols.union(set(sp.simplify(key).free_symbols))
        for value in start.values():
            free_symbols = free_symbols.union(set(sp.simplify(value).free_symbols))
        return free_symbols

    @batched("iterations")
    def limit(
        self,
        trajectory: dict,
        iterations: Batchable[int],
        start: dict,
        initial_values: Matrix | None = None,
        final_projection: Matrix | None = None,
    ) -> Batchable[Limit]:
        from ramanujantools import Limit

        def walk_function(iterations):
            return self.walk(trajectory, iterations, start)

        return Limit.walk_to_limit(
            iterations, walk_function, initial_values, final_projection
        )

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

    def eigenvals(self, poincare=False) -> dict:
        """
        Returns the eigenvalues of the matrix, which are the roots of the characteristic polynomials.

        Args:
            poincare: if True, returns the eigenvalues of the Poincare characteristic polynomial instead.
            False by default.
        """
        charpoly = self.charpoly(poincare)
        return sp.roots(charpoly)

    def sorted_eigenvals(self) -> list:
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

    def errors(self) -> list:
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
            q_reduced_list.append(sp.log(limit.as_rational().q).evalf(30))
        fit = np.polyfit(
            np.array(depths), np.array(q_reduced_list, dtype=np.float64), 1
        )
        return mp.mpf(fit[0])

    def kamidelta(self, depth=20) -> list[mp.mpf]:
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

    def at_infinity(self) -> Matrix:
        return Matrix(sp.Matrix(self).limit(n, sp.oo))

    def degrees(self, symbol: sp.Symbol = None) -> Matrix:
        r"""
        Returns a matrix of the degrees of each cell in the matrix.
        For a rational function $f = \frac{p}{q}$, the degree is defined as $deg(f) = deg(p) - deg(q)$.
        """
        if symbol is None:
            if len(self.free_symbols) != 1:
                raise ValueError(
                    f"Must specify symbol when matrix has more than one free symbol, got {self.free_symbols}"
                )
            symbol = list(self.free_symbols)[0]
        return Matrix(
            self.rows,
            self.cols,
            [
                sp.Poly(p, symbol).degree() - sp.Poly(q, symbol).degree()
                for p, q in (cell.as_numer_denom() for cell in self)
            ],
        )

    def jordan_form(self, calc_transform=True, **kwargs):
        """
        Overloads SymPy's jordan_form to automatically sort the Jordan blocks
        in descending order based on the absolute magnitude of the eigenvalues.
        """
        P, J = super().jordan_form(calc_transform=True, **kwargs)

        dim = self.shape[0]
        starts = [i for i in range(dim) if i == 0 or J[i - 1, i] == 0]
        ends = starts[1:] + [dim]
        blocks = [(J[s, s], P[:, s:e], J[s:e, s:e]) for s, e in zip(starts, ends)]

        def sort_key(b):
            try:
                return abs(complex(b[0].evalf()))
            except TypeError:
                return 0

        blocks.sort(key=sort_key, reverse=True)

        J_sorted = Matrix.diag(*[b[2] for b in blocks])

        if not calc_transform:
            return J_sorted

        P_sorted = Matrix.hstack(*[b[1] for b in blocks])
        return P_sorted, J_sorted

    def to_series_matrix(self, var: sp.Symbol, p: int, precision: int) -> SeriesMatrix:
        """
        Converts a symbolic matrix over a variable into a formal SeriesMatrix
        by expanding it around infinity using the ramification index p.
        """
        from ramanujantools.asymptotics.series_matrix import SeriesMatrix

        dim = self.shape[0]
        if not self.free_symbols:
            coeffs = [self] + [Matrix.zeros(dim, dim) for _ in range(precision - 1)]
            return SeriesMatrix(coeffs, p=p, precision=precision)

        t = sp.Symbol("t", positive=True)
        M_t = self.subs({var: t ** (-p)})

        expanded_matrix = M_t.applyfunc(
            lambda x: sp.series(x, t, 0, precision).removeO()
        )

        coeffs = []
        for i in range(precision):
            coeff_matrix = expanded_matrix.applyfunc(lambda x: sp.expand(x).coeff(t, i))

            if coeff_matrix.has(t) or coeff_matrix.has(var):
                raise ValueError(
                    f"Coefficient {i} failed to evaluate to a constant matrix."
                )
            coeffs.append(coeff_matrix)

        return SeriesMatrix(coeffs, p=p, precision=precision)

    @lru_cache
    def _get_reducer_at_precision(
        self, precision: int, force: bool = False
    ) -> "Reducer":
        """
        Executes a single, targeted Birkhoff-Trjitzinsky reduction at a specific precision.
        """
        from ramanujantools.asymptotics import Reducer

        var = sp.Symbol("n") if not self.free_symbols else list(self.free_symbols)[0]
        U = self.companion_coboundary_matrix(var)
        cvm_matrix = self.coboundary(U, var)

        reducer = Reducer.from_matrix(
            cvm_matrix.transpose(),
            precision=precision,
            force=force,  # Pass the bypass flag down to the engine
        )
        reducer.reduce()
        return reducer

    @lru_cache
    def _get_reducer(self, precision=None) -> "Reducer":
        """
        Pre-conditions the matrix via CVM and safely executes the precision
        backoff loop to return a fully solved Reducer instance and the CVM transformation matrix U.
        """
        from ramanujantools.asymptotics import Reducer

        if precision is not None:
            print(
                f"[DEBUG STABILITY] Explicit precision {precision} requested. Bypassing stability loop."
            )
            return self._get_reducer_at_precision(precision, force=True)

        var = sp.Symbol("n") if not self.free_symbols else list(self.free_symbols)[0]
        U = self.companion_coboundary_matrix(var)
        cvm_matrix = self.coboundary(U, var)

        degrees = [d for d in cvm_matrix.degrees(var) if d != -sp.oo]
        S = max(degrees) - min(degrees) if degrees else 1
        max_precision = (self.shape[0] ** 2) * max(S, 1) + self.shape[0]

        current_precision = self.shape[0]
        step_size = 1
        last_expr_matrix = None
        current_reducer = None

        print(f"\n[DEBUG STABILITY] Starting backoff. Max boundary: {max_precision}")

        while current_precision <= max_precision:
            try:
                print(f"[DEBUG STABILITY] Attempting precision: {current_precision}")
                reducer = self._get_reducer_at_precision(current_precision, force=False)

                growth_grid = reducer.canonical_growth_matrix()
                expr_matrix = Reducer._growth_to_expr_matrix(growth_grid, var)

                if last_expr_matrix is not None:
                    diff = (expr_matrix - last_expr_matrix).applyfunc(sp.expand)
                    if diff.is_zero_matrix:
                        print(
                            f"[DEBUG STABILITY] Math stabilized at precision {current_precision}! Output is solid."
                        )
                        return current_reducer

                print(
                    f"[DEBUG STABILITY] Math shifted. Stepping up to {current_precision + step_size}."
                )
                last_expr_matrix = expr_matrix
                current_reducer = reducer
                current_precision += step_size

            except Exception as e:
                required = getattr(e, "required_precision", current_precision + 1)
                print(
                    f"[DEBUG STABILITY] Starved! Error requested precision {required}."
                )
                current_precision = max(current_precision + 1, required)
                last_expr_matrix = None

        print(
            "[DEBUG STABILITY] Hit maximum ramification bound. Returning safest computed matrix."
        )
        return current_reducer

    @lru_cache
    def _asymptotic_growth_matrix(self, precision) -> list[list["GrowthRate"]]:
        from ramanujantools.asymptotics.growth_rate import GrowthRate

        free_syms = list(self.free_symbols)
        var = sp.Symbol("n") if not free_syms else free_syms[0]
        dim = self.shape[0]

        reducer = self._get_reducer(precision)
        U = self.companion_coboundary_matrix(var)
        U_inv = U.inv()

        # Print the Transformation Matrix
        print("\n[DEBUG TRANSLATION] U_inv matrix:")
        sp.pprint(U_inv)

        cvm_grid_T = reducer.canonical_growth_matrix()
        cvm_grid = list(map(list, zip(*cvm_grid_T)))

        # Print the CVM Grid (The formal solutions)
        print("\n[DEBUG TRANSLATION] CVM Grid (Before Multiplication):")
        for r_idx, r in enumerate(cvm_grid):
            expr_row = [c.as_expr(var) for c in r]
            print(f"  CVM Row {r_idx}: {expr_row}")

        physical_grid = []
        for row in range(dim):
            physical_row = []
            for col in range(dim):
                dominant_growth = GrowthRate()
                for k in range(dim):
                    u_val = U_inv[k, col]
                    if u_val != sp.S.Zero:
                        num, den = sp.numer(u_val), sp.denom(u_val)
                        degree_shift = sp.degree(num, var) - sp.degree(den, var)
                        u_growth = GrowthRate(
                            exp_base=sp.S.One,
                            polynomial_degree=sp.simplify(degree_shift),
                        )
                        dominant_growth += cvm_grid[row][k] * u_growth
                physical_row.append(dominant_growth)
            physical_grid.append(physical_row)

        # Print the final mixed physical grid
        print("\n[DEBUG TRANSLATION] Final Physical Grid:")
        for r_idx, r in enumerate(physical_grid):
            expr_row = [c.as_expr(var) for c in r]
            print(f"  Phys Row {r_idx}: {expr_row}")

        return physical_grid

    def canonical_fundamental_matrix(self, precision=None) -> Matrix:
        """
        Returns the Canonical Fundamental Matrix (CFM) of the linear system of difference equations defined by self,
        with regard to multiplication to the right: $M(0) * M(1) * ... * M(n-1)$.
        The CFM is defined as a formal set of solutions for the system such that they are asymptotically distinct.
        More documentation in Reducer.
        """
        from ramanujantools.asymptotics.reducer import Reducer

        free_syms = list(self.free_symbols)
        var = n if not free_syms else free_syms[0]

        # Fetch the mathematically pure grid of smart objects
        growth_matrix = self._asymptotic_growth_matrix(precision=precision)

        # Render the final human-readable expressions
        return Reducer._growth_to_expr_matrix(growth_matrix, var)

    def asymptotics(self, precision=None) -> list[sp.Expr]:
        """
        Returns the dominant asymptotic bounds for each physical variable in the system.

        This calculates the L_infinity norm equivalent for the rows of the Canonical
        Fundamental Matrix, safely filtering out sub-dominant trajectories to return
        the absolute upper bound of the sequence's growth at infinity.
        """
        var = sp.Symbol("n") if not self.free_symbols else list(self.free_symbols)[0]

        growth_grid = self._asymptotic_growth_matrix(precision=precision)
        bounds = []

        for row in growth_grid:
            # Safely find the dominant growth using your custom __gt__ operator
            dominant_growth = row[0]
            for current in row[1:]:
                if current > dominant_growth:
                    dominant_growth = current

            # The as_expr() method inherently handles the zero-base safety check
            bounds.append(dominant_growth.as_expr(var))

        return bounds
