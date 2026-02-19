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

        if not len(matrix.free_symbols) == 1:
            raise ValueError("Input matrix must depend on exactly one variable.")

        self.var = list(matrix.free_symbols)[0]
        self.precision = precision
        self.p = p
        self.dim = matrix.shape[0]

        self.factorial_power = max(matrix.degrees())
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
        The main state-machine loop. Runs until the system is fully diagonalized,
        then returns the extracted canonical data.
        """
        max_iterations = 10
        iterations = 0

        while not self.is_canonical and iterations < max_iterations:
            M0 = self.M.coeffs[0]
            P, J = M0.jordan_form()

            if J.is_diagonal():
                # Step 2: Distinct eigenvalues, block-diagonalize the tail
                self.split(P, J)
            else:
                # Step 3: Jordan blocks detected, apply Newton Polygon shearing
                self.shear()

            iterations += 1

        if not self.is_canonical:
            raise RuntimeError("Failed to reach canonical form within iteration limit.")

        return self.get_canonical_data()

    def split(self, P: Matrix, J: Matrix) -> None:
        """
        Executes the Splitting Lemma to block-diagonalize the system.
        Updates self.M and self.S_total in place.
        """
        S_step = SeriesMatrix([P], p=self.p, precision=self.precision)
        self.S_total = self.S_total * S_step
        self.M = S_step.inverse() * self.M * S_step

        for k in range(1, self.precision):
            R_k = self.M.coeffs[k]

            if R_k.is_diagonal():
                continue

            R_off = R_k - Matrix.diag(*[R_k[i, i] for i in range(self.dim)])

            # Eigenvalues are guaranteed to be distinct since J is diagonal here
            Y_mat = self._solve_sylvester_diagonal(J, -R_off)

            G_coeffs = (
                [Matrix.eye(self.dim)]
                + [Matrix.zeros(self.dim, self.dim)] * (k - 1)
                + [Y_mat]
            )
            G = SeriesMatrix(G_coeffs, p=self.p, precision=self.precision)

            self.S_total = self.S_total * G
            self.M = G.inverse() * self.M * G.shift()

        # If we reach the end of the precision tail, we are fully diagonalized
        self.is_canonical = True

    def shear(self) -> None:
        """Applies the Newton Polygon to handle Jordan blocks (Phase 3)."""
        raise NotImplementedError("Phase 3 (Shearing) logic goes here.")

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
