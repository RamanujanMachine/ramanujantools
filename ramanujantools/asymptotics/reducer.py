from __future__ import annotations

import sympy as sp

from ramanujantools import Matrix
from ramanujantools.asymptotics import SeriesMatrix


class Reducer:
    """
    Implements the Birkhoff-Trjitzinsky algorithm to compute the formal
    canonical fundamental matrix for linear difference systems.
    """

    def __init__(
        self,
        series: SeriesMatrix,
        var: sp.Symbol,
        factorial_power: int = 0,
        precision: int = 5,
        p: int = 1,
    ) -> None:
        """Strict constructor. Expects a pre-normalized SeriesMatrix."""
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
        self.children = []  # To hold our recursive sub-reducers
        self._is_reduced = False

    @classmethod
    def from_matrix(cls, matrix: Matrix, precision: int = 5, p: int = 1) -> Reducer:
        if not matrix.is_square():
            raise ValueError("Input matrix must be square.")

        free_syms = list(matrix.free_symbols)
        if len(free_syms) > 1:
            raise ValueError("Input matrix must depend on at most one variable.")

        dim = matrix.shape[0]
        var = sp.Symbol("n") if len(free_syms) == 0 else free_syms[0]
        factorial_power = max(matrix.degrees(var))

        normalized_matrix = matrix / (var**factorial_power)
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
        sys_mat, C_vec = sp.zeros(m * n, m * n), sp.zeros(m * n, 1)

        for j in range(n):
            for i in range(m):
                row_idx = j * m + i
                C_vec[row_idx, 0] = C[i, j]
                for k in range(m):  # A * X term
                    sys_mat[row_idx, j * m + k] += A[i, k]
                for k in range(n):  # -X * B term
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

    def reduce(self) -> Reducer:
        max_iterations = max(20, self.dim * 3)
        iterations = 0

        while not self._is_reduced and iterations < max_iterations:
            M0 = self.M.coeffs[0]
            if M0.is_zero_matrix:
                self.M = self.M.divide_by_t()
                self.factorial_power -= sp.Rational(1, self.p)
                continue

            k_target = self.M.get_first_non_scalar_index()
            if k_target is None:
                self._is_reduced = True
                break

            M_target = self.M.coeffs[k_target]
            P, J_target = M_target.jordan_form()

            self.S_total = self.S_total * SeriesMatrix(
                [P], p=self.p, precision=self.precision
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
        dim, blocks = self.dim, self._get_blocks(J_target)

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
                padded_G = (
                    [Matrix.eye(dim)] + [Matrix.zeros(dim, dim)] * (m - 1) + [Y_mat]
                )
                padded_G += [Matrix.zeros(dim, dim)] * (self.precision - len(padded_G))
                G = SeriesMatrix(padded_G, p=self.p, precision=self.precision)
                self.S_total, self.M = self.S_total * G, self.M.coboundary(G)

    def _compute_shear_slope(self) -> sp.Rational:
        """
        Constructs the exact Lower Convex Hull of the matrix valuations and returns
        the shearing slope 'g' (the steepest negative slope on the lower hull).
        """
        lambda_val = self.M.coeffs[0][0, 0]

        # Delegate the algebraic shift directly to the SeriesMatrix
        shifted_series = self.M.shift_leading_eigenvalue(lambda_val)
        vals = shifted_series.valuations()

        points = []
        for i in range(self.dim):
            for j in range(self.dim):
                v = vals[i, j]
                if v != sp.oo:
                    points.append((j - i, v))

        # Group by x, keeping only the lowest y for each vertical line
        lowest_points = {}
        for x, y in points:
            if x not in lowest_points or y < lowest_points[x]:
                lowest_points[x] = y

        sorted_x = sorted(lowest_points.keys())
        hull_points = [(x, lowest_points[x]) for x in sorted_x]

        # Build the exact Lower Convex Hull using a Monotone Chain
        lower_hull = []
        for p in hull_points:
            while len(lower_hull) >= 2:
                p1 = lower_hull[-2]
                p2 = lower_hull[-1]
                p3 = p

                # Calculate slopes between the last two segments
                slope1 = sp.Rational(p2[1] - p1[1], p2[0] - p1[0])
                slope2 = sp.Rational(p3[1] - p2[1], p3[0] - p2[0])

                # If the slope decreases or stays the same, the point p2 is an interior
                # point (not strictly convex) and must be discarded.
                if slope2 <= slope1:
                    lower_hull.pop()
                else:
                    break
            lower_hull.append(p)

        # The steepest negative slope is mathematically guaranteed to be
        # the very first segment of the lower convex hull!
        if len(lower_hull) < 2:
            return sp.S.Zero

        p1, p2 = lower_hull[0], lower_hull[1]
        steepest_slope = sp.Rational(p2[1] - p1[1], p2[0] - p1[0])

        # We return the positive scalar g
        g = -steepest_slope

        return max(sp.S.Zero, g)

    def shear(self) -> None:
        g = self._compute_shear_slope()

        if g == sp.S.Zero:
            # Solid bedrock! Block is unsplittable. Stop shearing and let extraction handle log(n).
            self._is_reduced = True
            return

        if not g.is_integer:
            g, b = g.as_numer_denom()
            self.M, self.S_total = self.M.ramify(b), self.S_total.ramify(b)
            self.p *= b
            self.precision *= b

        t = sp.Symbol("t", positive=True)
        S_sym = Matrix.diag(*[t ** (i * g) for i in range(self.dim)])

        # Use the updated classmethod
        S_series = self.__class__._symbolic_to_series(
            S_sym, self.var, self.p, self.precision, self.dim
        )

        self.S_total = self.S_total * S_series
        self.M = self.M.shear_coboundary(g)

    def canonical_data(self) -> tuple[sp.Number, Matrix, Matrix]:
        """
        Extracts the canonical growth matrices.
        Returns:
            factorial_power: The exponent d for the factorial growth (n!)^d.
            Lambda: The exponential growth base matrix (e^Q).
            D: The algebraic growth matrix (n^D).
        """
        if not self._is_reduced:
            self.reduce()

        Lambda = self.M.coeffs[0]

        # If precision is at least 2, we can extract D. Otherwise, D is 0.
        if self.precision > 1:
            M1 = self.M.coeffs[1]
            D = Lambda.inv() * M1
        else:
            D = Matrix.zeros(self.dim)

        return self.factorial_power, Lambda, D

    def asymptotic_expressions(self) -> list[sp.Expr]:
        # 1. Recursive Delegation
        if self.children:
            return [
                sol for child in self.children for sol in child.asymptotic_expressions()
            ]

        if not self._is_reduced:
            self.reduce()

        d, n, t = self.factorial_power, self.var, sp.Symbol("t", positive=True)
        solutions, jordan_depth = [], 0

        for i in range(self.dim):
            lambda_val = self.M.coeffs[0][i, i]
            if lambda_val == sp.S.Zero:
                solutions.append(sp.S.Zero)
                continue

            # Logarithmic Trigger for permanent chains
            is_jordan_link = any(
                self.M.coeffs[k][i - 1, i] != sp.S.Zero for k in range(self.precision)
            )
            jordan_depth = jordan_depth + 1 if (i > 0 and is_jordan_link) else 0

            L_t = sp.S.One
            max_k = min(self.precision, self.p + 1)
            for k in range(1, max_k):
                L_t += (self.M.coeffs[k][i, i] / lambda_val) * (t**k)

            log_series = sp.series(sp.log(L_t), t, 0, self.p + 1)
            Q_n, D_val = sp.S.Zero, sp.S.Zero

            for k in range(1, self.p + 1):
                c_k = log_series.coeff(t, k)
                if c_k == sp.S.Zero:
                    continue

                if k < self.p:
                    power = 1 - sp.Rational(k, self.p)
                    Q_n += (c_k / power) * (n**power)
                elif k == self.p:
                    D_val = c_k

            expr = (sp.factorial(n) ** d) * (lambda_val**n) * sp.exp(Q_n) * (n**D_val)

            # Inject the log(n)
            if jordan_depth > 0:
                expr = expr * (sp.log(n) ** jordan_depth)

            solutions.append(expr)

        return solutions
