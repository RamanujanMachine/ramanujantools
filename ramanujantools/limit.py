from __future__ import annotations

from typing import Callable

import sympy as sp
import mpmath as mp

from ramanujantools import IntegerRelation, Matrix


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


class Limit:
    r"""
    Represents a mathematical limit of a `walk` operation.

    Contains two matrices for the two last steps of calculation.
    Uses the last step to extract constants, and the previous one to determine precision.
    """

    def __init__(
        self,
        current: Matrix,
        previous: Matrix,
        p_vectors: list[Matrix] | None = None,
        q_vectors: list[Matrix] | None = None,
    ):
        self.current = current
        self.previous = previous
        self.p_vectors = p_vectors or [
            Matrix.e(self.N(), 0, column=False),
            Matrix.e(self.N(), self.N() - 1, column=True),
        ]

        self.q_vectors = q_vectors or [
            Matrix.e(self.N(), 1, column=False),
            Matrix.e(self.N(), self.N() - 1, column=True),
        ]

    def __repr__(self) -> str:
        return f"Limit({self.current}, {self.previous})"

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Limit) -> bool:
        return (
            self.current == other.current
            and self.previous == other.previous
            and self.p_vectors == other.p_vectors
            and self.q_vectors == other.q_vectors
        )

    def N(self) -> int:
        return self.current.rows

    @property
    def mp(self):
        limit_ctx = mp.mp.clone()
        limit_ctx.dps = max(self.precision() * 1.1, 15)
        return limit_ctx

    @staticmethod
    def walk_to_limit(
        iterations: list[int],
        walk_function: Callable[list[int], list[Limit]],
        p_vectors: list[Matrix] | None = None,
        q_vectors: list[Matrix] | None = None,
    ) -> list[Limit]:
        previous_values = [depth - 1 for depth in iterations]
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
                    p_vectors,
                    q_vectors,
                )
            )
        return limits

    def as_rational(self, previous=False) -> list:
        r"""
        Returns the limit as a rational number $\frac{p}{q}$.

        The numbers p and q are extracted using matrix multiplication.
        For each number, a row vector `v1` and a column vector `v2` are received.
        We then extract a number using `v1 * M * v2`, where M is the limit matrix.

        Researcher's note: rational representation of the limit is so far only well-defined for 2x2 matrices,
        and we are still looking for a generalization of this representation for NxN matrices.
        Args:
            previous: Will use $M$ = `self.previous` if True, else `self.current`. False by default.
        Returns:
            A list of the form [p, q], representing the rational number.
        """
        matrix = self.previous if previous else self.current
        p = sp.Rational((self.p_vectors[0] * matrix * self.p_vectors[1])[0])
        q = sp.Rational((self.q_vectors[0] * matrix * self.q_vectors[1])[0])
        return [
            sp.Integer(p.numerator * q.denominator),
            sp.Integer(p.denominator * q.numerator),
        ]

    def precision(self, base: int = 10) -> int:
        """
        Returns the error in 'digits' for the PCF convergence.

        Args:
            base: The numerical base in which to return the precision (by default 10)
        """
        p1, q1 = self.as_rational()
        p2, q2 = self.as_rational(previous=True)
        numerator = p1 * q2 - q1 * p2
        denominator = q1 * q2
        if denominator == 0:
            return 0
        if numerator == 0:
            return 100  # big enough, this should be infinity
        # extracting real because sometimes log returns a complex with tiny imaginary type due to precision
        digits = -mp.re((mp.log(int(numerator), base) - mp.log(int(denominator), base)))
        return int(mp.floor(digits))

    def as_float(self) -> mp.mpf:
        r"""
        Returns the limit as a floating point number f, such that $m \cdot v = f$, where `m=self` and `v=vector`.
        """
        p, q = self.as_rational()
        return self.mp.mpf(p) / self.mp.mpf(q)

    def as_rounded_number(self) -> str:
        """
        Same as `as_float`, but also rounds the result to the shortest number possible within the error range.
        """
        return most_round_in_range(self.as_float(), 10 ** -self.precision())

    def delta(self, L: mp.mpf) -> mp.mpf:
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{{q_n}^{1+\delta}}$
        Args:
            L: $L$
        Returns:
            the delta value as defined above.
        """
        p, q = self.as_rational()
        gcd = sp.gcd(p, q)
        reduced_q = self.mp.fabs(q // gcd)
        if q == 0:
            return self.mp.mpf("-inf")
        if reduced_q == 1:
            return self.mp.mpf("inf")
        return -(1 + self.mp.log(self.mp.fabs(L - (p / q)), reduced_q))

    def coefficients_from_pslq(self, pslq_results, active_indices):
        coefficients = [0] * self.N()
        for i in active_indices:
            coefficients[i] = pslq_results.pop(0)
        return coefficients

    def identify_rational(
        self, column_index=-1, maxcoeff=1000
    ) -> IntegerRelation | None:
        r"""
        Searches for constants $a_0, \dots, a_{N-1}
        such that $0 \approx \prod_{i=0}^{N-1}a_i * p_i$,
        Where $p_i$ a column of the Limit matrix.

        This is essentially the same as `self.identify(0, column_inde)`

        Args:
            column_index: The column to use in order to extract $p_i$. -1 by default.
        Returns:
            a string describing the integer relation, if exists. None otherwise.
        """
        pslq_result = self.mp.pslq(self.current.col(column_index), maxcoeff=maxcoeff)
        if pslq_result is None:
            return None
        result = IntegerRelation(
            [self.coefficients_from_pslq(pslq_result, range(self.N()))]
        )
        return result

    def identify(
        self, L: mp.mpf, column_index=-1, maxcoeff=1000
    ) -> IntegerRelation | None:
        r"""
        Given a constant $L$, searches for constants $a_0, \dots, a_{N-1}, b_0, \dots, b_{N-1}$
        such that $0 \approx \prod_{i=0}^{N-1}a_i * p_i - L * \prod_{i=0}^{N-1}b_i * p_i$,
        Where $p_i$ a column of the Limit matrix.

        Args:
            column_index: The column to use in order to extract $p_i$. -1 by default.
        Returns:
            a string describing the integer relation, if exists. None otherwise.
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
        numerator = self.coefficients_from_pslq(
            pslq_result[:total_indices], used_indices
        )
        denominator = self.coefficients_from_pslq(
            pslq_result[total_indices:], used_indices
        )
        result = IntegerRelation([numerator, denominator])
        self.set_vectors(result)
        return result

    def set_vectors(self, relation: IntegerRelation) -> None:
        self.p_vectors = [Matrix([relation.coefficients[0]]), Matrix.e(self.N(), -1)]
        self.q_vectors = [Matrix([relation.coefficients[1]]), -Matrix.e(self.N(), -1)]
