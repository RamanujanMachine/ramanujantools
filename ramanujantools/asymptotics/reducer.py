from __future__ import annotations

import sympy as sp

from ramanujantools import Matrix
from ramanujantools.asymptotics import SeriesMatrix


class Reducer:
    """
    Implements the Birkhoff-Trjitzinsky algorithm to compute the formal
    canonical fundamental matrix for linear difference systems.
    """

    def __init__(self, matrix: Matrix, precision: int = 5, p: int = 1) -> None:
        if not matrix.is_square():
            raise ValueError("Input matrix must be square.")

        free_syms = list(matrix.free_symbols)
        if len(free_syms) > 1:
            raise ValueError("Input matrix must depend on at most one variable.")

        self.precision = precision
        self.p = p
        self.dim = matrix.shape[0]

        self.var = sp.Symbol("n") if len(free_syms) == 0 else free_syms[0]
        self.factorial_power = max(matrix.degrees(self.var))
        normalized_matrix = matrix / (self.var**self.factorial_power)

        self.M = self._symbolic_to_series(normalized_matrix)

        # The accumulated global gauge transformation S(n)
        self.S_total = SeriesMatrix(
            [Matrix.eye(self.dim)], p=self.p, precision=self.precision
        )

        self.is_canonical = False

    def _symbolic_to_series(self, matrix: Matrix) -> SeriesMatrix:
        """
        Expands a symbolic matrix M(n) at n=oo into a formal series in t = n^(-1/p).
        """
        if not matrix.free_symbols:
            coeffs = [matrix] + [
                Matrix.zeros(self.dim, self.dim) for _ in range(self.precision - 1)
            ]
            return SeriesMatrix(coeffs, p=self.p, precision=self.precision)

        t = sp.Symbol("t", positive=True)
        M_t = matrix.subs({self.var: t ** (-self.p)})

        coeffs = []
        for i in range(self.precision):
            coeff_matrix = M_t.applyfunc(
                lambda x: sp.series(x, t, 0, self.precision).coeff(t, i)
            )

            if coeff_matrix.has(t) or coeff_matrix.has(self.var):
                raise ValueError(
                    f"Coefficient {i} failed to evaluate to a constant matrix."
                )

            coeffs.append(coeff_matrix)

        return SeriesMatrix(coeffs, p=self.p, precision=self.precision)

    @staticmethod
    def _solve_sylvester_diagonal(J: Matrix, R: Matrix) -> Matrix:
        """
        Solves the Sylvester equation: J*Y - Y*J = R for Y.
        Assumption: J is a diagonal matrix with DISTINCT eigenvalues.
        """
        rows, cols = J.shape
        Y = Matrix.zeros(rows, cols)

        eigenvalues = [J[i, i] for i in range(rows)]

        for i in range(rows):
            for j in range(cols):
                if i == j:
                    continue

                diff = eigenvalues[i] - eigenvalues[j]
                if diff == sp.S.Zero:
                    # We hit duplicate roots. Simple scalar division won't work.
                    raise NotImplementedError(
                        "Duplicate eigenvalues detected! Block Sylvester solver required."
                    )

                Y[i, j] = R[i, j] / diff

        return Y

    def reduce(self) -> tuple[sp.Number, Matrix, Matrix]:
        """
        The main state-machine loop. Runs until the system is fully diagonalized.
        """
        max_iterations = max(20, self.dim * 3)
        iterations = 0

        while not self.is_canonical and iterations < max_iterations:
            M0 = self.M.coeffs[0]

            if M0.is_zero_matrix:
                self.M = self.M.divide_by_t()
                self.factorial_power -= 1
                continue

            k_target = self.M.get_first_non_scalar_index()

            if k_target is None:
                # If every single matrix in the tail is scalar, the system is fully decoupled!
                self.is_canonical = True
                break

            M_target = self.M.coeffs[k_target]
            P, J_target = M_target.jordan_form()

            S_step = SeriesMatrix([P], p=self.p, precision=self.precision)
            self.S_total = self.S_total * S_step
            self.M = self.M.similarity_transform(P, J_target if k_target == 0 else None)

            if J_target.is_diagonal():
                self.split(k_target, J_target)
            else:
                self.shear()

            iterations += 1

        if not self.is_canonical:
            raise RuntimeError("Failed to reach canonical form within iteration limit.")

        return self.get_canonical_data()

    def split(self, k_target: int, J_target: Matrix) -> None:
        """
        Executes the generalized Splitting Lemma.
        Uses the first non-scalar matrix J_target (at t^k_target)
        to block-diagonalize the higher-order tail.
        """
        for m in range(1, self.precision - k_target):
            target_idx = k_target + m
            R_k = self.M.coeffs[target_idx]

            if R_k.is_diagonal():
                continue

            R_off = R_k - Matrix.diag(*[R_k[i, i] for i in range(self.dim)])
            Y_mat = self._solve_sylvester_diagonal(J_target, -R_off)

            G_coeffs = (
                [Matrix.eye(self.dim)]
                + [Matrix.zeros(self.dim, self.dim)] * (m - 1)
                + [Y_mat]
            )
            G = SeriesMatrix(G_coeffs, p=self.p, precision=self.precision)

            self.S_total = self.S_total * G
            self.M = self.M.coboundary(G)

        self.is_canonical = True

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
        """
        Applies the Newton Polygon shearing transformation to split nilpotent Jordan blocks,
        ramifying the system if fractional Puiseux powers are required.
        """
        g = self._compute_shear_slope()

        if g == sp.S.Zero:
            raise NotImplementedError(
                "Permanent Jordan block detected! Exponential extraction for "
                "regular singularities is not yet fully implemented."
            )

        if not g.is_integer:
            g, b = g.as_numer_denom()

            self.M = self.M.ramify(b)
            self.S_total = self.S_total.ramify(b)

            self.p *= b
            self.precision *= b

        t = sp.Symbol("t", positive=True)

        S_sym = Matrix.diag(*[t ** (i * g) for i in range(self.dim)])
        S_series = self._symbolic_to_series(S_sym)
        self.S_total = self.S_total * S_series
        self.M = self.M.shear_coboundary(g)

    def get_canonical_data(self) -> tuple[sp.Number, Matrix, Matrix]:
        """
        Extracts the canonical growth matrices.
        Returns:
            factorial_power: The exponent d for the factorial growth (n!)^d.
            Lambda: The exponential growth base matrix (e^Q).
            D: The algebraic growth matrix (n^D).
        """
        if not self.is_canonical:
            raise RuntimeError("System is not canonical yet. Call reduce() first.")

        Lambda = self.M.coeffs[0]

        # If precision is at least 2, we can extract D. Otherwise, D is 0.
        if self.precision > 1:
            M1 = self.M.coeffs[1]
            D = Lambda.inv() * M1
        else:
            D = Matrix.zeros(self.dim)

        return self.factorial_power, Lambda, D

    def get_asymptotic_expressions(self) -> list[sp.Expr]:
        """
        Converts the canonical matrices into concrete SymPy expressions
        representing the asymptotic growth of each fundamental solution.
        The returned list strictly preserves the diagonal order of the canonical matrices.
        """
        if not self.is_canonical:
            raise RuntimeError("System is not canonical yet. Call reduce() first.")

        if self.p > 1:
            raise NotImplementedError(
                "Translating ramified (p > 1) systems back to scalar expressions "
                "requires formal exponential integration."
            )

        d, Lambda, D = self.get_canonical_data()
        n = self.var

        solutions = []
        for i in range(self.dim):
            lambda_val = Lambda[i, i]
            d_val = D[i, i]

            # u_i(n) = (n!)^d * (lambda_i)^n * n^{D_i}
            expr = (sp.factorial(n) ** d) * (lambda_val**n) * (n**d_val)
            solutions.append(expr)

        return solutions
