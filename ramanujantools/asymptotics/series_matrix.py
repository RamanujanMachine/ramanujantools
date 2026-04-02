from __future__ import annotations

import sympy as sp
from sympy.abc import t, n

from ramanujantools import Matrix


class SeriesMatrix:
    r"""
    Represents a formal power series (or Puiseux series) with matrix coefficients.

    The series is expanded in terms of a local parameter $t = n^{-1/p}$, taking the form:
    $$M(t) = A_0 + A_1 t + A_2 t^2 + \dots + A_k t^k + \mathcal{O}(t^{k+1})$$

    This class provides the core algebraic ring operations (addition, Cauchy multiplication,
    formal inversion) and gauge transformations (shifts, shears, coboundaries) required
    for the Birkhoff-Trjitzinsky reduction algorithm.
    """

    def __init__(self, coeffs, p=1, precision=None):
        """
        Constructs a SeriesMatrix from a list of matrix coefficients,
        with optional ramification `p` such that $t = n^{-1/p}$.

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

    def __mul__(self, other: "SeriesMatrix") -> "SeriesMatrix":
        """
        Computes the Cauchy product of two Formal Power Series matrices.
        Automatically bounds the result to the lowest precision of the two operands.
        """
        if self.p != other.p:
            raise ValueError(
                "SeriesMatrix ramification indices (p) must match for multiplication."
            )

        # Mathematically, the product of O(t^A) and O(t^B) is valid up to O(t^min(A, B))
        out_precision = min(self.precision, other.precision)
        new_coeffs = [Matrix.zeros(*self.shape) for _ in range(out_precision)]

        for i in range(out_precision):
            if self.coeffs[i].is_zero_matrix:
                continue

            for j in range(out_precision - i):
                if other.coeffs[j].is_zero_matrix:
                    continue

                new_coeffs[i + j] += self.coeffs[i] * other.coeffs[j]

        return SeriesMatrix(new_coeffs, p=self.p, precision=out_precision).simplify()

    def inverse(self) -> SeriesMatrix:
        r"""
        Computes the formal inverse series $V(t) = S(t)^{-1}$.

        By definition, $S(t) \cdot V(t) = I$. Expanding this into a Cauchy product
        and equating the coefficients for $t^k$ yields the recurrence relation:
        $$\sum_{i=0}^{k} S_i V_{k-i} = 0 \quad \text{for } k > 0$$

        Isolating the $k$-th coefficient $V_k$ provides the explicit update rule used
        by this method:
        $$V_k = -S_0^{-1} \sum_{i=1}^{k} S_i V_{k-i}$$
        """
        V_coeffs = [Matrix.zeros(*self.shape) for _ in range(self.precision)]

        V_0 = self.coeffs[0].inv()
        V_coeffs[0] = V_0

        for k in range(1, self.precision):
            sum_terms = Matrix.zeros(*self.shape)
            for i in range(1, k + 1):
                sum_terms += self.coeffs[i] * V_coeffs[k - i]

            # DEFLATE + CRUSH ALGEBRA: Prevent Y^6 from swelling with un-evaluated roots
            V_coeffs[k] = (-V_0 * sum_terms).applyfunc(
                lambda x: sp.cancel(sp.expand(x))
            )

        return SeriesMatrix(V_coeffs, p=self.p, precision=self.precision)

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

    def shift(self) -> SeriesMatrix:
        r"""
        Applies the discrete shift operator $n \to n + 1$ to the formal series.

        Given the local parameter $t = n^{-1/p}$, substituting $n+1$ yields the new parameter:
        $$t_{\text{new}} = (n + 1)^{-1/p} = (t^{-p} + 1)^{-1/p} = t(1 + t^p)^{-1/p}$$

        Applying this substitution to the $m$-th term of the series ($A_m t^m$) and expanding
        it using the generalized binomial theorem gives:
        $$A_m t_{\text{new}}^m = A_m t^m (1 + t^p)^{-m/p} = \sum_{j=0}^{\infty} A_m \binom{-m/p}{j} t^{m + pj}$$

        This method computes this expansion up to the defined precision and accumulates
        the shifted coefficients.
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
        """
        Factors out a power of t from the entire series.
        Mathematically equivalent to M(t) / t. This physically shifts all matrix
        coefficients one index to the left and pads the tail with a zero matrix.
        """
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

    def coboundary(self, T: SeriesMatrix) -> SeriesMatrix:
        """
        Computes the right-acting discrete coboundary T(n+1)^{-1} * M(n) * T(n).
        Assumes T is an invertible formal power series (det(T_0) != 0).
        """
        T_shifted = T.shift()
        T_inv = T_shifted.inverse()
        left_mult = T_inv * self
        res = left_mult * T
        return res

    def truncate(self, new_precision: int) -> SeriesMatrix:
        """Sheds trailing precision terms to optimize performance."""
        if new_precision >= self.precision:
            return self
        return SeriesMatrix(
            self.coeffs[:new_precision], p=self.p, precision=new_precision
        )

    def _shear_row_corrections(self, g: int) -> list[list[sp.Expr]]:
        """Pre-computes the generalized binomial coefficients for the row shifts."""
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
        return row_corrections

    def shear_coboundary(
        self, g: sp.Rational | int, target_precision: int | None = None
    ) -> tuple[SeriesMatrix, int]:
        """
        Applies a shearing transformation S(t) to the series to expose sub-exponential
        growth, where $S(t) = diag(1, t^g, t^{2g}, \\dots)$.

        Args:
            g: The shear slope.
            target_precision: If provided, truncates the resulting series to this length,
                              saving heavy CAS simplification on discarded tail terms.

        Returns:
            A tuple containing the sheared SeriesMatrix and the integer `h` representing
            the overall degree shift (used to adjust the global factorial power).
        """
        row_corrections = self._shear_row_corrections(g)
        power_dict = {}

        for m in range(self.precision):
            for i in range(self.shape[0]):
                for j in range(self.shape[0]):
                    val_M = self.coeffs[m][i, j]
                    if val_M == sp.S.Zero:
                        continue

                    shift = int((j - i) * g)
                    for c in range(self.precision):
                        val_C = row_corrections[i][c]
                        if val_C == sp.S.Zero:
                            continue

                        power = m + c + shift
                        if power not in power_dict:
                            power_dict[power] = Matrix.zeros(*self.shape)
                        power_dict[power][i, j] += val_C * val_M

        min_power = min(power_dict.keys()) if power_dict else 0
        h = -min_power

        output_precision = (
            target_precision if target_precision is not None else self.precision
        )
        new_coeffs = []
        for k in range(output_precision):
            target_power = k - h
            if target_power in power_dict:
                new_coeffs.append(power_dict[target_power])
            else:
                new_coeffs.append(Matrix.zeros(*self.shape))

        return SeriesMatrix(
            new_coeffs, p=self.p, precision=output_precision
        ).simplify(), h

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
        Scans the series to find the index $k$ of the first matrix $A_k$ that is not a scalar matrix
        (i.e., not a multiple of the identity matrix).

        In the Birkhoff-Trjitzinsky algorithm, if the entire series consists only of
        scalar matrices, the system is fundamentally reduced and the algorithm terminates.
        """
        for k, C in enumerate(self.coeffs):
            # A matrix is scalar if it's diagonal and has <= 1 unique diagonal elements
            if C.is_diagonal() and len(set(C.diagonal())) <= 1:
                continue
            return k

        return None

    def simplify(self) -> SeriesMatrix:
        def crush(x):
            return sp.cancel(sp.expand(x))

        new_coeffs = [c.applyfunc(crush) for c in self.coeffs]

        return type(self)(new_coeffs, p=self.p, precision=self.precision)

    @classmethod
    def from_matrix(
        cls, matrix: Matrix, var: sp.Symbol, p: int, precision: int
    ) -> SeriesMatrix:
        """
        Converts a normalized symbolic matrix into a formal SeriesMatrix at infinity
        by executing a formal Taylor expansion: substituting n = t^(-p) and extracting coefficients.
        """
        dim = matrix.shape[0]

        if not matrix.free_symbols:
            coeffs = [matrix] + [Matrix.zeros(dim, dim) for _ in range(precision - 1)]
            return cls(coeffs, p=p, precision=precision)

        M_t = matrix.subs({var: t ** (-p)})

        expanded_matrix = M_t.applyfunc(
            lambda x: sp.series(x, t, 0, precision).removeO()
        )

        coeffs = []
        for i in range(precision):
            coeff_matrix = expanded_matrix.applyfunc(lambda x: sp.expand(x).coeff(t, i))
            if coeff_matrix.has(t) or coeff_matrix.has(var):
                raise ValueError(f"Coefficient {i} failed to evaluate to a constant.")
            coeffs.append(coeff_matrix)

        return cls(coeffs, p=p, precision=precision)
