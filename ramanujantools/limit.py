from __future__ import annotations

from typing import Callable

import sympy as sp
import mpmath as mp

from ramanujantools import Matrix


def first_unmatch(a: str, b: str) -> int:
    """
    Returns the index of the biggest digit that does not match in a and b
    """
    size = min(len(a), len(b))
    for i in range(size):
        if a[i] != b[i]:
            return i
    return size


def round_attempt(lower_bound: mp.mpf, upper_bound: mp.mpf) -> str:
    lower = str(lower_bound)
    upper = str(upper_bound)
    up_to = first_unmatch(lower, upper) + 1
    return upper[0:up_to]


def most_round_in_range(num: mp.mpf, err: mp.mpf) -> str:
    return min(round_attempt(num, num + err), round_attempt(num - err, num), key=len)


def coefficients_from_pslq(pslq_results: list[int], active_indices: list[int], N: int):
    coefficients = [0] * N
    for i in active_indices:
        coefficients[i] = pslq_results.pop(0)
    return coefficients


class Limit:
    r"""
    Represents a mathematical limit of a `walk` operation.

    Contains two matrices for the two last steps of calculation.
    Uses the last step to extract constants, and the previous one to determine precision.
    The diagonal of the 2x2 matrix calculated as $IV \cdot M \cdot FP$ consists of the p and q values
    that define the limit as $p/q$.
    """

    def __init__(
        self,
        current: Matrix,
        previous: Matrix,
        initial_values: Matrix | None = None,
        final_projection: Matrix | None = None,
    ):
        self.current = current
        self.previous = previous
        self.initial_values = (
            initial_values
            or Matrix.hstack(
                Matrix.e(self.N(), 0),
                Matrix.e(self.N(), 1),
            ).transpose()
        )
        self.final_projection = final_projection or Matrix.hstack(
            Matrix.e(self.N(), -1),
            Matrix.e(self.N(), -1),
        )

    def __repr__(self) -> str:
        return f"Limit({self.current}, {self.previous}, {self.initial_values}, {self.final_projection})"

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Limit) -> bool:
        return (
            self.current == other.current
            and self.previous == other.previous
            and self.initial_values == other.initial_values
            and self.final_projection == other.final_projection
        )

    def N(self) -> int:
        """
        Returns N, the dimension of the square limit matrix.
        """
        return self.current.rows

    def p_vectors(self) -> list[Matrix]:
        """
        Returns the IV row vector and the FP column vector used to calculate p.
        """
        return [self.initial_values.row(0), self.final_projection.col(0)]

    def q_vectors(self) -> list[Matrix]:
        """
        Returns the IV row vector and the FP column vector used to calculate q.
        """
        return [self.initial_values.row(1), self.final_projection.col(1)]

    def p(self, previous=False) -> sp.Rational:
        """
        Returns p, the numerator of the rational limit.

        Args:
            previous: if True, will use `self.previous` matrix to calculate p, else `self.current`. False by default.
        """
        matrix = self.previous if previous else self.current
        row, col = self.p_vectors()
        return (row * matrix * col)[0]

    def q(self, previous=False) -> sp.Rational:
        """
        Returns q, the numerator of the rational limit.

        Args:
            previous: if True, will use `self.previous` matrix to calculate q, else `self.current`. False by default.
        """
        matrix = self.previous if previous else self.current
        row, col = self.q_vectors()
        return (row * matrix * col)[0]

    @property
    def mp(self):
        """
        Returns an mpmath context with the precision set to 15 digits.
        """
        limit_ctx = mp.mp.clone()
        limit_ctx.dps = max(self.precision() * 1.1, 15)
        return limit_ctx

    @staticmethod
    def walk_to_limit(
        iterations: list[int],
        walk_function: Callable[list[int], list[Limit]],
        initial_values: Matrix | None = None,
        final_projection: Matrix | None = None,
    ) -> list[Limit]:
        """
        Utility function that receives a walk function logic and a list of iterations,
        and returns a list of Limit objects for these iterations

        Args:
            iterations: A list of the required iterations to calculate the limits for.
            walk_function: The walk logic function.
            initial_values: The initial values matrix, defaults to $e_1$ and $e_2$.
            final_projection: The final projection matrix, defaults to $e_{-1}$ and $e_{-1}$.
        """
        previous_values = [max(depth - 1, 0) for depth in iterations]
        walk_iterations = sorted(list(set(iterations + previous_values)))
        walk_matrices = walk_function(walk_iterations)

        current_index = 0
        limits = []
        for depth in iterations:
            while depth != walk_iterations[current_index]:
                current_index += 1
            limits.append(
                Limit(
                    walk_matrices[current_index],
                    walk_matrices[current_index - 1],
                    initial_values,
                    final_projection,
                )
            )
        return limits

    def as_rational(self, previous=False) -> sp.Rational:
        r"""
        Returns the limit as a rational number $\frac{p}{q}$.

        Args:
            previous: Will use $M$ = `self.previous` if True, else `self.current`. False by default.
        """
        return sp.Rational(self.p(previous), self.q(previous))

    def precision(self, base: int = 10) -> int:
        """
        Returns the error in 'digits' for the PCF convergence.

        Args:
            base: The numerical base in which to return the precision (by default 10)
        """
        diff = self.as_rational(previous=False) - self.as_rational(previous=True)
        if not diff.is_rational:  # division by 0, etc
            return 0
        if diff == 0:
            return 100  # big enough, this should be infinity
        # extracting real because sometimes log returns a complex with tiny imaginary type due to precision
        digits = -mp.re(mp.log(diff, base))
        return int(mp.floor(digits))

    def as_float(self) -> mp.mpf:
        r"""
        Returns the limit as a floating point number f, such that $m \cdot v = f$, where `m=self` and `v=vector`.
        """
        return self.mp.mpf(self.as_rational())

    def as_rounded_number(self) -> str:
        """
        Same as `as_float`, but also rounds the result to the shortest number possible within the error range.
        """
        return most_round_in_range(self.as_float(), 10 ** -self.precision())

    def delta(self, L: mp.mpf) -> mp.mpf:
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p}{q} - L| = \frac{1}{{q}^{1+\delta}}$
        Args:
            L: $L$
        Returns:
            the delta value as defined above.
        """
        p, q = self.as_rational().as_numer_denom()
        if q == 0:
            return self.mp.mpf("-inf")
        if q == 1:
            return self.mp.mpf("inf")
        return -(1 + self.mp.log(self.mp.fabs(L - (p / q)), q))

    def identify_rational(self, column_index=-1, maxcoeff=1000) -> Matrix | None:
        r"""
        Searches for a vector $c$ to match a column $p$ of the limit matrix,
        such that $c \cdot p \approx 0$,

        This is essentially the same as `self.identify(0, column_index, maxcoeff)`
        Args:
            column_index: The column index that defines $p$. -1 by default (rightmost column).
        Returns:
            A 1-row matrix $c$ such that $c \cdot p \approx 0$.
        """
        pslq_result = self.mp.pslq(self.current.col(column_index), maxcoeff=maxcoeff)
        if pslq_result is None:
            return None
        return Matrix([coefficients_from_pslq(pslq_result, range(self.N()), self.N())])

    def identify(self, L: mp.mpf, column_index=-1, maxcoeff=1000) -> Matrix | None:
        r"""
        Given a constant $L$, searches for two vectors $a, b$
        such that $\frac{a \cdot p}{b \cdot p} \approx L$,
        Where $p$ a column of the limit matrix.

        Args:
            column_index: The column index that defines $p$. -1 by default (rightmost column).
        Returns:
            A 2-row initial values matrix $m$ (made from rows $a$ and $b$),
            such that $\frac{a \cdot p}{b \cdot p} \approx L$.
        """
        if L == 0:
            return self.identify_rational()

        def linear_independent_indices():
            indices = list(range(self.N()))

            def remove_index(pslq_result):
                for index, value in reversed(list(enumerate(pslq_result))):
                    if value != 0:  # found a dependency
                        indices.pop(index)
                        return

            while len(indices) > 1:  # a single index is linear independent vacuously
                integer_sequences = [self.current.col(column_index)[i] for i in indices]
                pslq_result = self.mp.pslq(integer_sequences, maxcoeff=maxcoeff)
                if pslq_result is None:
                    return indices
                remove_index(pslq_result)

            return indices

        used_indices = linear_independent_indices()
        total_indices = len(used_indices)
        integer_sequences = [self.current.col(column_index)[i] for i in used_indices]
        to_identify = integer_sequences + [p * L for p in integer_sequences]
        pslq_result = self.mp.pslq(to_identify, maxcoeff=maxcoeff, maxsteps=maxcoeff)
        if pslq_result is None:
            return None
        # We normalize the sign so that the first coefficient is positive,
        # to avoid ambiguity as results are equivalent up to the sign.
        numerator = Matrix(
            coefficients_from_pslq(pslq_result[:total_indices], used_indices, self.N())
        )
        denominator = Matrix(
            coefficients_from_pslq(pslq_result[total_indices:], used_indices, self.N())
        )
        sign = sp.sign(numerator[0])
        initial_values = Matrix.hstack(
            sign * numerator, -sign * denominator
        ).transpose()
        self.initial_values = initial_values
        self.final_projection = Matrix.hstack(
            Matrix.e(self.N(), column_index),
            Matrix.e(self.N(), column_index),
        )
        return initial_values
