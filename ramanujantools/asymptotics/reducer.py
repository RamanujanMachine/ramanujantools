from __future__ import annotations

from functools import lru_cache

import sympy as sp


from ramanujantools import Matrix
from ramanujantools.asymptotics import (
    GrowthRate,
    SeriesMatrix,
    ShearOverflowError,
    EigenvalueBlindnessError,
    RowNullityError,
    InputTruncationError,
)


class Reducer:
    """
    Implements the Birkhoff-Trjitzinsky algorithm to compute the formal
    canonical fundamental matrix for linear difference systems.

    Sources:
        - Analytic Theory of Singular Difference Equations: George D Birkhoff and Waldemar J Trjitzinsky
        - Resurrecting the Asymptotics of Linear Recurrences: Jet Wimp and Doron Zeilberger
        - Galois theory of difference equations, chapter 7.2: Marius van der Put and Michael Singer
    """

    def __init__(
        self,
        series: SeriesMatrix,
        var: sp.Symbol,
        factorial_power: int = 0,
        precision: int = 5,
        p: int = 1,
    ) -> None:
        """
        Initializes the Reducer with a pre-conditioned formal power series.
        Usually called internally by `Reducer.from_matrix()`.
        """
        self.M = series
        self.var = var
        self.factorial_power = factorial_power
        self.precision = precision
        self.p = p
        self.dim = series.coeffs[0].shape[0]
        self.S_total = SeriesMatrix(
            [Matrix.eye(self.dim)], p=self.p, precision=self.precision
        )
        self._is_reduced = False
        self.children = []

    @classmethod
    def from_matrix(cls, matrix: Matrix, precision: int = 5, p: int = 1) -> Reducer:
        """
        Initializes the Birkhoff-Trjitzinsky engine from a standard linear system matrix.
        Normalizes the matrix to isolate the factorial growth bound before converting
        it into a formal Taylor series.
        """
        if not matrix.is_square():
            raise ValueError("Input matrix must be square.")

        free_syms = list(matrix.free_symbols)
        if len(free_syms) > 1:
            raise ValueError("Input matrix must depend on at most one variable.")

        dim = matrix.shape[0]
        var = sp.Symbol("n") if len(free_syms) == 0 else free_syms[0]
        factorial_power = max(matrix.degrees(var))

        normalized_matrix = matrix / (var**factorial_power)
        required_precision = (
            -min(
                [
                    factorial_power
                    for factorial_power in normalized_matrix.degrees(var)
                    if factorial_power > -sp.oo
                ],
                default=0,
            )
            * p
            + 1
        )
        if precision < required_precision:
            raise InputTruncationError(
                required_precision=required_precision,
                message=f"Input Truncation! The deepest term requires a minimum precision of {required_precision}.",
            )

        series = cls._symbolic_to_series(normalized_matrix, var, p, precision, dim)

        return cls(
            series=series,
            var=var,
            factorial_power=factorial_power,
            precision=precision,
            p=p,
        )

    @classmethod
    def _symbolic_to_series(
        cls, matrix: Matrix, var: sp.Symbol, p: int, precision: int, dim: int
    ) -> SeriesMatrix:
        if not matrix.free_symbols:
            coeffs = [matrix] + [Matrix.zeros(dim, dim) for _ in range(precision - 1)]
            return SeriesMatrix(coeffs, p=p, precision=precision)

        t = sp.Symbol("t", positive=True)
        M_t = matrix.subs({var: t ** (-p)})

        coeffs = []
        for i in range(precision):
            coeff_matrix = M_t.applyfunc(
                lambda x: sp.series(x, t, 0, precision).coeff(t, i)
            )
            if coeff_matrix.has(t) or coeff_matrix.has(var):
                raise ValueError(
                    f"Coefficient {i} failed to evaluate to a constant matrix."
                )
            coeffs.append(coeff_matrix)

        return SeriesMatrix(coeffs, p=p, precision=precision)

    @staticmethod
    def _solve_sylvester(A: Matrix, B: Matrix, C: Matrix) -> Matrix:
        """Solves the Sylvester equation: A*X - X*B = C for X using Kronecker flattening."""
        m, n = A.shape[0], B.shape[0]
        sys_mat, C_vec = Matrix.zeros(m * n, m * n), Matrix.zeros(m * n, 1)

        for j in range(n):
            for i in range(m):
                row_idx = j * m + i
                C_vec[row_idx, 0] = C[i, j]
                for k in range(m):
                    sys_mat[row_idx, j * m + k] += A[i, k]
                for k in range(n):
                    sys_mat[row_idx, k * m + i] -= B[k, j]

        vec_X = sys_mat.LUsolve(C_vec)

        X = Matrix.zeros(m, n)
        for j in range(n):
            for i in range(m):
                X[i, j] = vec_X[j * m + i, 0]
        return X

    def _get_blocks(self, J_target: Matrix) -> list[tuple[int, int, sp.Expr]]:
        """Finds the boundaries (start_idx, end_idx, eigenvalue) of independent blocks."""
        blocks = []
        if self.dim == 0:
            return blocks
        current_eval, start_idx = J_target[0, 0], 0

        for i in range(1, self.dim):
            if J_target[i, i] != current_eval:
                blocks.append((start_idx, i, current_eval))
                current_eval, start_idx = J_target[i, i], i

        blocks.append((start_idx, self.dim, current_eval))
        return blocks

    @lru_cache
    def reduce(self) -> Reducer:
        """
        The core Birkhoff-Trjitzinsky reduction loop.
        Iteratively applies block diagonalization (split) or ramified shears
        until the matrix reaches a terminal canonical form.
        """
        max_iterations = max(20, self.dim * 3)
        iterations = 0
        zeros_shifted = 0

        while not self._is_reduced and iterations < max_iterations:
            M0 = self.M.coeffs[0]

            if M0.is_zero_matrix:
                if zeros_shifted >= self.precision:
                    raise ValueError(
                        f"Series exhausted after {zeros_shifted} shifts. Precision too low."
                    )

                self.M = self.M.divide_by_t()
                self.factorial_power -= sp.Rational(1, self.p)
                zeros_shifted += 1
                iterations += 1
                continue

            k_target = self.M.get_first_non_scalar_index()
            if k_target is None:
                self._is_reduced = True
                break

            M_target = self.M.coeffs[k_target]
            P, J_target = M_target.jordan_form()
            self.S_total = self.S_total * SeriesMatrix(
                [P], p=self.p, precision=self.S_total.precision
            )
            self.M = self.M.similarity_transform(P, J_target if k_target == 0 else None)
            unique_evals = list(
                dict.fromkeys([J_target[i, i] for i in range(self.dim)])
            )

            if len(unique_evals) > 1:
                self.split(k_target, J_target)
                blocks = self._get_blocks(J_target)
                for s_idx, e_idx, _ in blocks:
                    sub_coeffs = [
                        self.M.coeffs[k][s_idx:e_idx, s_idx:e_idx]
                        for k in range(self.precision)
                    ]
                    sub_series = SeriesMatrix(
                        sub_coeffs, p=self.p, precision=self.precision
                    )
                    sub_reducer = Reducer(
                        series=sub_series,
                        var=self.var,
                        factorial_power=self.factorial_power,
                        precision=self.precision,
                        p=self.p,
                    )
                    sub_reducer.reduce()
                    self.children.append(sub_reducer)

                self._is_reduced = True
                return self
            else:
                self.shear()

            iterations += 1

        if not self._is_reduced:
            raise RuntimeError("Failed to reach canonical form within iteration limit.")
        return self

    def split(self, k_target: int, J_target: Matrix) -> None:
        """
        Performs block diagonalization.
        When a leading coefficient matrix has distinct eigenvalues, this method uses
        Sylvester equations to decouple the system into independent smaller Jordan blocks.
        """
        dim, blocks = self.dim, self._get_blocks(J_target)

        max_sub_dim = max((e - s) for s, e, _ in blocks)
        buffer_needed = 0 if max_sub_dim == 1 else (max_sub_dim * max_sub_dim)
        needed_precision = self.p + 1 + buffer_needed

        if self.precision > needed_precision:
            self.M = self.M.truncate(needed_precision)
            self.precision = needed_precision

        for m in range(1, self.precision - k_target):
            R_k, Y_mat, needs_gauge = (
                self.M.coeffs[k_target + m],
                Matrix.zeros(dim, dim),
                False,
            )

            for s_i, e_i, eval_i in blocks:
                J_ii = J_target[s_i:e_i, s_i:e_i]
                for s_j, e_j, eval_j in blocks:
                    if eval_i == eval_j:
                        continue

                    J_jj, R_ij = J_target[s_j:e_j, s_j:e_j], R_k[s_i:e_i, s_j:e_j]

                    if not R_ij.is_zero_matrix:
                        needs_gauge = True
                        Y_ij = self._solve_sylvester(J_ii, J_jj, -R_ij)

                        for r in range(e_i - s_i):
                            for c in range(e_j - s_j):
                                Y_mat[s_i + r, s_j + c] = Y_ij[r, c]

            if needs_gauge:
                Y_mat = Y_mat.applyfunc(lambda x: sp.cancel(sp.radsimp(sp.cancel(x))))

                padded_G = (
                    [Matrix.eye(dim)] + [Matrix.zeros(dim, dim)] * (m - 1) + [Y_mat]
                )
                padded_G += [Matrix.zeros(dim, dim)] * (self.precision - len(padded_G))

                G = SeriesMatrix(padded_G, p=self.p, precision=self.precision)

                # SEVERED LINK: We DO NOT multiply self.S_total * G.
                # G is a near-identity matrix; it cannot affect the leading tail.
                self.M = self.M.coboundary(G)

                self.M = SeriesMatrix(
                    [
                        c.applyfunc(lambda x: sp.cancel(sp.expand(x)))
                        for c in self.M.coeffs
                    ],
                    p=self.p,
                    precision=self.precision,
                )

    def _compute_shear_slope(self) -> sp.Rational:
        """
        Calculates the steepest valid slope for a shear transformation by constructing
        a Newton Polygon from the valuation matrix of the shifted series.
        """
        exp_base = self.M.coeffs[0][0, 0]
        shifted_series = self.M.shift_leading_eigenvalue(exp_base)
        vals = shifted_series.valuations()

        points = []
        for i in range(self.dim):
            for j in range(self.dim):
                v = vals[i, j]
                if v != sp.oo:
                    points.append((j - i, v))

        lowest_points = {}
        for x, y in points:
            if x not in lowest_points or y < lowest_points[x]:
                lowest_points[x] = y

        sorted_x = sorted(lowest_points.keys())
        hull_points = [(x, lowest_points[x]) for x in sorted_x]

        lower_hull = []
        for p in hull_points:
            while len(lower_hull) >= 2:
                p1 = lower_hull[-2]
                p2 = lower_hull[-1]
                p3 = p

                slope1 = sp.Rational(p2[1] - p1[1], p2[0] - p1[0])
                slope2 = sp.Rational(p3[1] - p2[1], p3[0] - p2[0])

                if slope2 <= slope1:
                    lower_hull.pop()
                else:
                    break
            lower_hull.append(p)

        if len(lower_hull) < 2:
            return sp.S.Zero

        p1, p2 = lower_hull[0], lower_hull[1]
        steepest_slope = sp.Rational(p2[1] - p1[1], p2[0] - p1[0])
        g = -steepest_slope

        return max(sp.S.Zero, g)

    def _check_shear_overflow(self, g: sp.Rational | int) -> None:
        """
        Detects if a shear transformation will push non-zero terms past the
        allocated precision buffer of S_total.
        Uses a Global Maximum check to account for column-mixing by P matrices.
        """
        # Find the absolute deepest term ANYWHERE in the matrix
        global_deepest_k = 0
        for k in range(self.S_total.precision):
            if not self.S_total.coeffs[k].is_zero_matrix:
                global_deepest_k = k

        # Find the heaviest possible shift applied to any column
        # For a shear matrix S = diag(1, t^g, t^2g, ...), the max shift is on the last column
        max_shift = (self.dim - 1) * g

        # Check if the worst-case scenario breaks the boundary
        if global_deepest_k + max_shift >= self.S_total.precision:
            required_precision = int(global_deepest_k + max_shift + 1)
            outer_required = int(sp.ceiling(required_precision / self.p))

            raise ShearOverflowError(
                required_precision=outer_required,
                message=f"Global Shear Overflow! Deepest matrix term is at index {global_deepest_k}. "
                f"Max upcoming shift is {global_deepest_k + max_shift}, "
                f"S_total precision is only {self.S_total.precision}.",
            )

    def _check_eigenvalue_blindness(self, exp_base: sp.Expr) -> None:
        """
        Detects if the matrix is completely nilpotent at the current precision.
        """
        if exp_base == sp.S.Zero:
            raise EigenvalueBlindnessError(
                required_precision=self.precision + self.dim,
                message="Zero Eigenvalue Drop! System is completely nilpotent at current precision.",
            )

    def _check_cfm_validity(self, grid: list[list["GrowthRate"]]) -> None:
        """
        Checks that no physical variable can completely vanish.
        If an entire row is 0, a critical coupling term was starved of precision.
        """
        for row in range(self.dim):
            # A cell is algebraically zero if its base eigenvalue (exp_base) is 0
            if all(cell.exp_base == sp.S.Zero for cell in grid[row]):
                raise RowNullityError(
                    required_precision=self.precision + self.dim,
                    message=f"Row Nullity Violation! Physical variable at row {row} vanished completely.",
                )

    def shear(self) -> None:
        """
        Applies a ramification and shear transformation.
        Used when the leading matrix is nilpotent, this shifts the polynomial degrees
        of the variables to expose the hidden sub-exponential growths.
        """
        g = self._compute_shear_slope()

        if g == sp.S.Zero:
            self._check_eigenvalue_blindness(self.M.coeffs[0][0, 0])
            self._is_reduced = True
            return

        if not g.is_integer:
            g, b = g.as_numer_denom()
            self.M, self.S_total = self.M.ramify(b), self.S_total.ramify(b)
            self.p *= b
            self.precision *= b

        self._check_shear_overflow(g)

        t = sp.Symbol("t", positive=True)
        S_sym = Matrix.diag(*[t ** (i * g) for i in range(self.dim)])
        S_series = Reducer._symbolic_to_series(
            S_sym, self.var, self.p, self.S_total.precision, self.dim
        )

        self.S_total = self.S_total * S_series

        self.M, h = self.M.shear_coboundary(g)

        if h != 0:
            self.factorial_power += sp.Rational(h, self.p)

    def asymptotic_growth(self) -> list[GrowthRate | None]:
        """
        Extracts the raw, unmapped asymptotic components of the internal canonical basis.
        Returns a list of strongly-typed GrowthRate objects.
        """
        if not self._is_reduced:
            self.reduce()

        if self.children:
            return [sol for child in self.children for sol in child.asymptotic_growth()]

        factorial_power, n, t = (
            self.factorial_power,
            self.var,
            sp.Symbol("t", positive=True),
        )
        growths, log_power = [], 0

        for i in range(self.dim):
            exp_base = sp.cancel(sp.expand(self.M.coeffs[0][i, i]))
            self._check_eigenvalue_blindness(exp_base)

            is_jordan_link = False
            if i > 0 and exp_base == sp.cancel(
                sp.expand(self.M.coeffs[0][i - 1, i - 1])
            ):
                is_jordan_link = any(
                    sp.cancel(sp.expand(self.M.coeffs[k][i - 1, i])) != sp.S.Zero
                    for k in range(self.precision)
                )
            log_power = log_power + 1 if is_jordan_link else 0

            x = sp.S.Zero
            max_k = min(self.precision, self.p + 1)
            for k in range(1, max_k):
                x += (self.M.coeffs[k][i, i] / exp_base) * (t**k)

            log_series = sp.S.Zero
            for j in range(1, self.p + 1):
                log_series += ((-1) ** (j + 1) / sp.Rational(j)) * (x**j)

            log_series = sp.expand(log_series)

            sub_exp, polynomial_degree = sp.S.Zero, sp.S.Zero

            for k in range(1, self.p + 1):
                c_k = log_series.coeff(t, k)
                if c_k == sp.S.Zero:
                    continue

                c_k = sp.cancel(sp.expand(c_k))

                if k < self.p:
                    power = 1 - sp.Rational(k, self.p)
                    sub_exp += (c_k / power) * (n**power)
                elif k == self.p:
                    polynomial_degree = c_k

            growths.append(
                GrowthRate(
                    exp_base=exp_base,
                    sub_exp=sub_exp,
                    polynomial_degree=polynomial_degree,
                    log_power=log_power,
                    factorial_power=factorial_power,
                )
            )

        return growths

    def asymptotic_expressions(self) -> list[sp.Expr]:
        """
        Builds the 'classic' scalar expressions from the raw internal growth components.
        This perfectly preserves backward compatibility with older scalar tests.
        """
        return [
            g.as_expr(self.var) if g is not None else sp.S.Zero
            for g in self.asymptotic_growth()
        ]

    def canonical_growth_matrix(self) -> list[list[GrowthRate]]:
        r"""
        Constructs the 2D Canonical Fundamental Matrix (CFM) using the internal algebra
        of `GrowthRate` objects.

        This method maps the raw, 1D independent asymptotic solutions back into the
        physical coordinates of the original system by applying the accumulated
        gauge transformations $S_{\text{total}}(t)$.

        Mathematically, it computes the dominant asymptotic term for each cell in:
        $$Y(n) = S_{\text{total}}(n^{-1/p}) \cdot \text{diag}(E_1(n), \dots, E_N(n))$$

        For a specific physical variable (row) and independent solution (column), the
        gauge matrix $S_{\text{total}}$ shifts the polynomial degree of the solution based
        on its first non-zero Taylor coefficient at index $k$. The exact fractional shift
        applied to the polynomial degree is:
        $$\Delta D = -\frac{k}{p} - d$$
        where $p$ is the ramification index and $d$ is the global factorial power.

        Returns:
            A 2D list of `GrowthRate` objects. Each column $j$ represents an independent
            solution vector, and each row $i$ represents the asymptotic behavior of the
            $i$-th physical variable for that solution.
        """
        growths = self.asymptotic_growth()
        matrix_grid = []

        for row in range(self.dim):
            row_growths = []
            for i, g in enumerate(growths):
                cell_growth = GrowthRate()
                for k in range(self.S_total.precision):
                    coeff = self.S_total.coeffs[k][row, i]
                    if coeff != sp.S.Zero:
                        shift_growth = GrowthRate(
                            exp_base=sp.S.One,
                            polynomial_degree=-sp.Rational(k, self.p)
                            - g.factorial_power,
                        )

                        cell_growth = g * shift_growth
                        break

                row_growths.append(cell_growth)
            matrix_grid.append(row_growths)

        self._check_cfm_validity(matrix_grid)
        return matrix_grid

    @classmethod
    def _growth_to_expr_matrix(
        cls, growth_matrix: list[list[GrowthRate]], var: sp.Symbol
    ) -> Matrix:
        dim = len(growth_matrix)
        cfm = Matrix.zeros(dim, dim)

        for row in range(dim):
            for col in range(dim):
                cfm[row, col] = growth_matrix[row][col].as_expr(var)

        return cfm

    def canonical_fundamental_matrix(self) -> Matrix:
        """
        Converts the smart GrowthRate grid into a formal SymPy Matrix of expressions.
        This provides the final, human-readable fundamental solution set.
        """
        return Reducer._growth_to_expr_matrix(self.canonical_growth_matrix(), self.var)
