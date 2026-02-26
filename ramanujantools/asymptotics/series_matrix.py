from __future__ import annotations

import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix


class SeriesMatrix:
    def __init__(self, coeffs, p=1, precision=None):
        """
        Represents a formal matrix series: A_0 + A_1*t + A_2*t^2 + ...
        where t = n^(-1/p).

        Args:
            coeffs: List of Matrix objects [A_0, A_1, A_2, ...]
            p: The ramification index (integer). Default is 1 (t = 1/n).
            precision: Maximum number of terms to keep. Defaults to len(coeffs).
        """
        self.p = p
        self.precision = precision if precision is not None else len(coeffs)
        self.shape = coeffs[0].shape if coeffs else (0, 0)

        # Store coefficients up to precision, pad with zero matrices if needed
        self.coeffs = []
        for i in range(self.precision):
            if i < len(coeffs):
                if len(coeffs[i].free_symbols) > 0:
                    raise ValueError("SeriesMatrix can only receive numeric matrices!")
                self.coeffs.append(coeffs[i])
            else:
                self.coeffs.append(Matrix.zeros(*self.shape))

    def __add__(self, other) -> SeriesMatrix:
        if self.shape != other.shape or self.p != other.p:
            raise ValueError(
                "SeriesMatrix dimensions or ramification indices do not match."
            )
        new_precision = min(self.precision, other.precision)
        new_coeffs = [self.coeffs[i] + other.coeffs[i] for i in range(new_precision)]
        return SeriesMatrix(new_coeffs, p=self.p, precision=new_precision)

    def __mul__(self, other: SeriesMatrix) -> SeriesMatrix:
        """
        Computes the Cauchy product of two Formal Power Series matrices.
        """
        if self.p != other.p or self.precision != other.precision:
            raise ValueError("SeriesMatrix parameters must match for multiplication.")

        new_coeffs = [Matrix.zeros(*self.shape) for _ in range(self.precision)]

        for k in range(self.precision):
            coeff_sum = Matrix.zeros(*self.shape)
            # Standard discrete convolution: sum_{m=0}^k A_m * B_{k-m}
            for m in range(k + 1):
                coeff_sum += self.coeffs[m] * other.coeffs[k - m]
            new_coeffs[k] = coeff_sum

        return SeriesMatrix(new_coeffs, p=self.p, precision=self.precision)

    def __repr__(self) -> str:
        return (
            f"SeriesMatrix(shape={self.shape}, precision={self.precision}, p={self.p})"
        )

    def __str__(self) -> str:
        """Helper to see the series written out symbolically."""
        expr = Matrix.zeros(*self.shape)
        for i, coeff in enumerate(self.coeffs):
            expr += coeff * (n ** (-sp.Rational(i, self.p)))
        return str(expr)

    def inverse(self) -> SeriesMatrix:
        """
        Computes the formal inverse series V = S^(-1).
        S * V = I  =>  S_0*V_k + S_1*V_{k-1} + ... = 0
        V_k = -S_0^(-1) * (S_1*V_{k-1} + S_2*V_{k-2} + ...)
        """
        V_coeffs = [Matrix.zeros(*self.shape) for _ in range(self.precision)]

        V_0 = self.coeffs[0].inv()
        V_coeffs[0] = V_0

        for k in range(1, self.precision):
            sum_terms = Matrix.zeros(*self.shape)
            for i in range(1, k + 1):
                sum_terms += self.coeffs[i] * V_coeffs[k - i]
            V_coeffs[k] = -V_0 * sum_terms

        return SeriesMatrix(V_coeffs, p=self.p, precision=self.precision)

    def shift(self) -> SeriesMatrix:
        """
        The n -> n + 1 operator.
        Since t = n^(-1/p), substituting n -> n+1 means:
        t_new = (t^(-p) + 1)^(-1/p) = t * (1 + t^p)^(-1/p)
        We expand this using the generalized binomial theorem.
        """
        new_coeffs = [Matrix.zeros(*self.shape) for _ in range(self.precision)]

        for m in range(self.precision):
            A_m = self.coeffs[m]
            if A_m.is_zero_matrix:
                continue

            j = 0
            while True:
                k = m + self.p * j  # The new power of t
                if k >= self.precision:
                    break

                # Generalized binomial coefficient: (-m/p choose j)
                binom_coeff = sp.binomial(-sp.Rational(m, self.p), j)

                new_coeffs[k] += A_m * binom_coeff
                j += 1

        return SeriesMatrix(new_coeffs, p=self.p, precision=self.precision)

    def divide_by_t(self) -> SeriesMatrix:
        coeffs = self.coeffs[1:] + [Matrix.zeros(*self.shape)]
        return SeriesMatrix(coeffs, p=self.p, precision=self.precision)

    def valuations(self) -> Matrix:
        """
        Returns a matrix where each entry (i, j) is the valuation
        (the lowest power of t with a non-zero coefficient) of that cell.
        Returns sympy.oo (infinity) for cells that are strictly zero.
        """
        rows, cols = self.shape
        val_matrix = sp.zeros(rows, cols)

        for i in range(rows):
            for j in range(cols):
                val = sp.oo
                for k in range(self.precision):
                    if self.coeffs[k][i, j] != sp.S.Zero:
                        val = sp.Rational(k)
                        break
                val_matrix[i, j] = val

        return val_matrix

    def similarity_transform(self, P: Matrix, J: Matrix = None) -> SeriesMatrix:
        """
        Applies a constant similarity transformation P^{-1} * M(t) * P.
        If the series has no tail (is a constant matrix) and J is provided,
        it mathematically short-circuits the inversion.
        """
        has_tail = any(not C.is_zero_matrix for C in self.coeffs[1:])

        if not has_tail and J is not None:
            new_coeffs = [J] + [
                Matrix.zeros(*self.shape) for _ in range(self.precision - 1)
            ]
            return SeriesMatrix(new_coeffs, p=self.p, precision=self.precision)

        P_inv = P.inverse()
        new_coeffs = [P_inv * C * P for C in self.coeffs]

        return SeriesMatrix(new_coeffs, p=self.p, precision=self.precision)

    def coboundary(self, T: "SeriesMatrix") -> "SeriesMatrix":
        """
        Computes the right-acting discrete coboundary T(n+1)^{-1} * M(n) * T(n).
        Assumes T is an invertible formal power series (det(T_0) != 0).
        """
        return T.shift().inverse() * self * T

    def shear_coboundary(self, g: int) -> "SeriesMatrix":
        """
        Computes the exact discrete coboundary S(n+1)^{-1} * M(n) * S(n)
        for a diagonal shear matrix S = diag(1, t^g, t^{2g}, ...).

        This bypasses singular matrix inversion by analytically fusing the
        algebraic Laurent shift t^{(j-i)g} and the discrete Taylor correction
        (1 + t^p)^{ig/p} into a single, highly efficient array pass.
        """
        import sympy as sp
        from ramanujantools import Matrix

        row_corrections = []
        for i in range(self.shape[0]):
            exponent = sp.Rational(i * g, self.p)
            coeffs = [sp.S.Zero] * self.precision

            for k in range(self.precision // self.p):
                bin_coeff = sp.S.One
                for j in range(k):
                    bin_coeff *= (exponent - j) / sp.Rational(j + 1)

                idx = k * self.p
                if idx < self.precision:
                    coeffs[idx] = bin_coeff
            row_corrections.append(coeffs)

        new_coeffs = [Matrix.zeros(*self.shape) for _ in range(self.precision)]

        for k in range(self.precision):
            for i in range(self.shape[0]):
                for j in range(self.shape[0]):
                    val = sp.S.Zero
                    shift = int((j - i) * g)

                    # We need m + shift + c = k  =>  m = k - c - shift
                    for c in range(k + 1):
                        m = k - c - shift
                        if 0 <= m < self.precision:
                            val += row_corrections[i][c] * self.coeffs[m][i, j]

                    new_coeffs[k][i, j] = val

        return SeriesMatrix(new_coeffs, p=self.p, precision=self.precision)

    def shift_leading_eigenvalue(self, lambda_val: sp.Expr) -> SeriesMatrix:
        """
        Shifts the formal series by subtracting a scalar matrix from the leading term.
        Mathematically equivalent to M(t) - lambda_val * I.
        """
        new_coeffs = list(self.coeffs)
        # Shift only the M_0 coefficient by the eigenvalue identity matrix
        new_coeffs[0] = new_coeffs[0] - lambda_val * sp.eye(self.shape[0])

        return SeriesMatrix(new_coeffs, p=self.p, precision=self.precision)

    def ramify(self, b: int) -> SeriesMatrix:
        """
        Executes Phase 4: Substitutes t = tau^b, effectively spreading the coefficients
        to handle fractional powers (Puiseux series).
        Returns a new SeriesMatrix with a multiplied ramification index and precision.
        """
        new_precision = self.precision * b
        new_coeffs = [Matrix.zeros(*self.shape) for _ in range(new_precision)]

        for k in range(self.precision):
            new_coeffs[k * b] = self.coeffs[k]

        return SeriesMatrix(new_coeffs, p=self.p * b, precision=new_precision)

    def get_first_non_scalar_index(self) -> int | None:
        """
        Scans the series and returns the index of the first non-scalar matrix.
        A matrix is scalar if it is diagonal and all diagonal entries are identical.
        Returns None if the entire series consists of scalar matrices.
        """
        for k, C in enumerate(self.coeffs):
            # A matrix is scalar if it's diagonal and has <= 1 unique diagonal elements
            if C.is_diagonal() and len(set(C.diagonal())) <= 1:
                continue
            return k

        return None
