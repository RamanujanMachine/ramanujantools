from __future__ import annotations
from typing import List, Callable, Union

import sympy as sp
from mpmath import mp

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


def round_attempt(original: mp.mpf, with_error: mp.mpf) -> str:
    original = str(original)
    with_error = str(with_error)
    up_to = first_unmatch(original, with_error) + 1
    return with_error[0:up_to]


def most_round_in_range(num: mp.mpf, err: mp.mpf) -> str:
    return min(round_attempt(num, num + err), round_attempt(num, num - err), key=len)


class Limit:
    r"""
    Represents a mathematical limit of a `walk` operation.

    Contains two matrices for the two last steps of calculation.
    Uses the last step to extract constants, and the previous one to determine precision.
    """

    def __init__(self, current: Matrix, previous: Matrix):
        self.current = current
        self.previous = previous

    def __repr__(self) -> str:
        return f"Limit({self.current}, {self.previous})"

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Limit) -> bool:
        return self.current == other.current and self.previous == other.previous

    def N(self) -> int:
        return self.current.rows

    @staticmethod
    def walk_to_limit(
        iterations: List[int], walk_function: Callable[List[int], List[Limit]]
    ) -> List[Limit]:
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
                )
            )
        return limits

    def as_rational(
        self,
        p_vectors: Union[List[Matrix], type(None)] = None,
        q_vectors: Union[List[Matrix], type(None)] = None,
        previous=False,
    ) -> List:
        r"""
        Returns the limit as a rational number $\frac{p}{q}$.

        The numbers p and q are extracted using matrix multiplication.
        For each number, a row vector `v1` and a column vector `v2` are received.
        We then extract a number using `v1 * M * v2`, where M is the limit matrix.

        Researcher's note: rational representation of the limit is so far only well-defined for 2x2 matrices,
        and we are still looking for a generalization of this representation for NxN matrices.
        Args:
            p_vectors: The extraction vectors for the numerator $p$.
            q_vectors: The extraction vectors for the denominator $q$.
            previous: Will use $M$ = `self.previous` if True, else `self.current`. False by default.
        Returns:
            A list of the form [p, q], representing the rational number.
        """
        p_vectors = p_vectors or [
            Matrix.e(self.N(), 0, column=False),
            Matrix.e(self.N(), self.N() - 1, column=True),
        ]

        q_vectors = q_vectors or [
            Matrix.e(self.N(), 1, column=False),
            Matrix.e(self.N(), self.N() - 1, column=True),
        ]

        matrix = self.previous if previous else self.current
        p = sp.Rational((p_vectors[0] * matrix * p_vectors[1])[0])
        q = sp.Rational((q_vectors[0] * matrix * q_vectors[1])[0])
        return [
            sp.Integer(p.numerator * q.denominator),
            sp.Integer(p.denominator * q.numerator),
        ]

    def precision(
        self,
        p_vectors: Union[List[Matrix], type(None)] = None,
        q_vectors: Union[List[Matrix], type(None)] = None,
        base: int = 10,
    ) -> int:
        """
        Returns the error in 'digits' for the PCF convergence.

        Args:
            base: The numerical base in which to return the precision (by default 10)
        """
        try:
            p1, q1 = self.as_rational(p_vectors, q_vectors)
            p2, q2 = self.as_rational(p_vectors, q_vectors, previous=True)
            numerator = p1 * q2 - q1 * p2
            denominator = q1 * q2
            # extracting real because sometimes log returns a complex with tiny imaginary type due to precision
            digits = -mp.re(
                (mp.log(int(numerator), base) - mp.log(int(denominator), base))
            )
            return int(mp.floor(digits))
        except (ZeroDivisionError, ValueError):
            return 0

    def increase_precision(
        self,
        p_vectors: Union[List[Matrix], type(None)] = None,
        q_vectors: Union[List[Matrix], type(None)] = None,
    ) -> int:
        """
        Increases the global mpmath precision to the lever required to handle this limit.
        Returns the current precision after the increase.
        """
        requested_precision = (
            self.precision(p_vectors, q_vectors) * 1.1
        )  # Taking 10% digits buffer
        mp.dps = max(mp.dps, requested_precision)
        return mp.dps

    def as_float(
        self,
        p_vectors: Union[List[Matrix], type(None)] = None,
        q_vectors: Union[List[Matrix], type(None)] = None,
    ) -> mp.mpf:
        r"""
        Returns the limit as a floating point number f, such that $m \cdot v = f$, where `m=self` and `v=vector`.

        This function increases the global mpmath precision if needed.
        """
        self.increase_precision(p_vectors, q_vectors)
        p, q = self.as_rational(p_vectors, q_vectors)
        return mp.mpf(p) / mp.mpf(q)

    def as_rounded_number(
        self,
        p_vectors: Union[List[Matrix], type(None)] = None,
        q_vectors: Union[List[Matrix], type(None)] = None,
    ) -> str:
        """
        Same as `as_float`, but also rounds the result to the shortest number possible within the error range.

        This function increases the global mpmath precision if needed.
        """
        return most_round_in_range(
            self.as_float(p_vectors, q_vectors),
            10 ** -self.precision(p_vectors, q_vectors),
        )

    def delta(
        self,
        L: mp.mpf,
        p_vectors: Union[List[Matrix], type(None)] = None,
        q_vectors: Union[List[Matrix], type(None)] = None,
    ) -> mp.mpf:
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{q_n}^{1+\delta}$

        This function increases the global mpmath precision if needed.
        Args:
            L: $L$
        Returns:
            the delta value as defined above.
        """
        self.increase_precision(p_vectors, q_vectors)
        p, q = self.as_rational(p_vectors, q_vectors)
        gcd = sp.gcd(p, q)
        reduced_q = mp.fabs(q // gcd)
        if reduced_q == 1:
            return mp.mpf("inf")
        return -(1 + mp.log(mp.fabs(L - (p / q)), reduced_q))
