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
        max_iterations = 10
        iterations = 0

        while not self.is_canonical and iterations < max_iterations:
            M0 = self.M.coeffs[0]
            P, J = M0.jordan_form()

            if M0.is_zero_matrix:
                self.M = self.M.divide_by_t()
                self.factorial_power -= 1
                continue

            # Align the entire series with the Jordan basis of M0
            # Since P is a constant matrix, its shift is just itself.
            S_step = SeriesMatrix([P], p=self.p, precision=self.precision)
            self.S_total = self.S_total * S_step
            self.M = S_step.inverse() * self.M * S_step

            if J.is_diagonal():
                # Step 2: Distinct eigenvalues, clean the tail
                self.split(J)
            else:
                # Step 3: Jordan blocks detected, apply Newton Polygon shearing
                self.shear()

            iterations += 1

        if not self.is_canonical:
            raise RuntimeError("Failed to reach canonical form within iteration limit.")

        return self.get_canonical_data()

    def split(self, J: Matrix) -> None:
        """
        Executes the Splitting Lemma to block-diagonalize the tail.
        """
        for k in range(1, self.precision):
            R_k = self.M.coeffs[k]

            if R_k.is_diagonal():
                continue

            R_off = R_k - Matrix.diag(*[R_k[i, i] for i in range(self.dim)])

            Y_mat = self._solve_sylvester_diagonal(J, -R_off)

            G_coeffs = (
                [Matrix.eye(self.dim)]
                + [Matrix.zeros(self.dim, self.dim)] * (k - 1)
                + [Y_mat]
            )
            G = SeriesMatrix(G_coeffs, p=self.p, precision=self.precision)

            self.S_total = self.S_total * G
            self.M = G.inverse() * self.M * G.shift()

        self.is_canonical = True

    def _compute_shear_slope(self) -> sp.Rational:
        """
        Constructs the Newton Polygon from the matrix valuations and returns
        the shearing slope 'g' (the steepest negative slope on the lower hull).
        """
        vals = self.M.valuations()

        # 1. Create the points (x = j - i, y = valuation)
        points = []
        for i in range(self.dim):
            for j in range(self.dim):
                v = vals[i, j]
                if v != sp.oo:
                    points.append((j - i, v))

        # 2. Group by x, keeping only the lowest y for each vertical line
        lowest_points = {}
        for x, y in points:
            if x not in lowest_points or y < lowest_points[x]:
                lowest_points[x] = y

        sorted_x = sorted(lowest_points.keys())
        hull_points = [(x, lowest_points[x]) for x in sorted_x]

        # 3. Find the steepest negative slope (which yields the maximum positive g)
        max_g = sp.S.Zero

        for p1 in hull_points:
            for p2 in hull_points:
                x1, y1 = p1
                x2, y2 = p2

                if x1 < x2:
                    # slope = (y2 - y1) / (x2 - x1)
                    # g = -slope = (y1 - y2) / (x2 - x1)
                    g = (y1 - y2) / sp.Rational(x2 - x1)
                    if g > max_g:
                        max_g = g

        return max_g

    def shear(self) -> None:
        """
        Executes Phase 3: Applies the Newton Polygon shearing transformation
        to split nilpotent Jordan blocks.
        """
        g = self._compute_shear_slope()

        if g == sp.S.Zero:
            raise NotImplementedError(
                "Permanent Jordan block detected! Exponential extraction for "
                "regular singularities is not yet fully implemented."
            )

        if not g.is_integer:
            raise NotImplementedError(
                f"Fractional slope g={g} detected! Phase 4 (Ramification) is required."
            )

        g = int(g)
        t = sp.Symbol("t", positive=True)

        # 1. Update the global gauge receipt.
        S_sym = Matrix.diag(*[t ** (i * g) for i in range(self.dim)])
        S_series = self._symbolic_to_series(S_sym)
        self.S_total = self.S_total * S_series

        # 2. Tell the series matrix to execute the shear analytically
        self.M = self.M.apply_diagonal_shear(g)

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

        return self.factorial_power, Lambda.simplify(), D.simplify()
