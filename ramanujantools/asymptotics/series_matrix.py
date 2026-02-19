from __future__ import annotations

import sympy as sp

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
        assert self.shape == other.shape and self.p == other.p
        new_precision = min(self.precision, other.precision)
        new_coeffs = [self.coeffs[i] + other.coeffs[i] for i in range(new_precision)]
        return SeriesMatrix(new_coeffs, p=self.p, precision=new_precision)

    def __mul__(self, other) -> SeriesMatrix:
        """Cauchy product of two series. O(K^2) matrix multiplications."""
        assert self.shape[1] == other.shape[0] and self.p == other.p
        new_precision = min(self.precision, other.precision)
        new_coeffs = [
            Matrix.zeros(self.shape[0], other.shape[1]) for _ in range(new_precision)
        ]

        for k in range(new_precision):
            for i in range(k + 1):
                new_coeffs[k] += self.coeffs[i] * other.coeffs[k - i]

        return SeriesMatrix(new_coeffs, p=self.p, precision=new_precision)

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

    def __repr__(self) -> str:
        return (
            f"SeriesMatrix(shape={self.shape}, precision={self.precision}, p={self.p})"
        )
