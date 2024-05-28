from __future__ import annotations
from typing import List
import math

from mpmath import mp

from ramanujan import Matrix


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


class Limit(Matrix):
    r"""
    Represents a mathematical limit of a `walk` operation.
    """

    def __repr__(self) -> str:
        matrix_string = repr(Matrix(self)).replace("Matrix(", "")[:-1]
        return f"Limit({matrix_string})"

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Limit) -> bool:
        """
        Returns true iff two limits converge to the same value.
        """
        p, q = self.as_rational()
        other_p, other_q = other.as_rational()
        return p * other_q == q * other_p

    def precision(self, p_indices=[0, -1], q_indices=[1, -1], base: int = 10) -> int:
        """
        Returns the error in 'digits' for the PCF convergence.

        Args:
            base: The numerical base in which to return the precision (by default 10)
        """
        diff = abs(
            mp.mpq(*(self.as_rational(p_indices, q_indices)))
            - mp.mpq(
                *(
                    self.as_rational(
                        [p_indices[0], p_indices[1] - 1],
                        [q_indices[0], q_indices[1] - 1],
                    )
                )
            )
        )
        return int(mp.floor(-mp.log(diff, base)))

    def increase_precision(self, p_indices=[0, -1], q_indices=[1, -1]) -> int:
        """
        Increases the global mpmath precision to the lever required to handle this limit.
        Returns the current precision after the increase.
        """
        requested_precision = (
            self.precision(p_indices, q_indices) * 1.1
        )  # Taking 10% digits buffer
        mp.dps = max(mp.dps, requested_precision)
        return mp.dps

    def as_rational(self, p_indices=[0, -1], q_indices=[1, -1]) -> List:
        r"""
        Returns the limit as a rational number $\frac{p}{q}$.

        Researcher's note: rational representation of the limit is so far only well-defined for 2x2 matrices,
        and we are still looking for a generalization of this representation for NxN matrices.
        Args:
            p_indices: the indices of the numerator $p$
            q_indices: the indices of the denominator $q$
        Returns:
            a list of the form [p, q], representing the rational number.
        """
        return [self[p_indices[0], p_indices[1]], self[q_indices[0], q_indices[1]]]

    def as_float(self, p_indices=[0, -1], q_indices=[1, -1]) -> mp.mpf:
        r"""
        Returns the limit as a floating point number f, such that $m \cdot v = f$, where `m=self` and `v=vector`.

        This function increases the global mpmath precision if needed.
        """
        self.increase_precision(p_indices, q_indices)
        p, q = self.as_rational(p_indices, q_indices)
        return mp.mpf(p) / mp.mpf(q)

    def as_rounded_number(self, p_indices=[0, -1], q_indices=[1, -1]) -> str:
        """
        Same as `as_float`, but also rounds the result to the shortest number possible within the error range.

        This function increases the global mpmath precision if needed.
        """
        return most_round_in_range(
            self.as_float(p_indices, q_indices),
            10 ** -self.precision(p_indices, q_indices),
        )

    def delta(self, L: mp.mpf, p_indices=[0, -1], q_indices=[1, -1]) -> mp.mpf:
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{q_n}^{1+\delta}$

        This function increases the global mpmath precision if needed.
        Args:
            L: $L$
        Returns:
            the delta value as defined above.
        """
        self.increase_precision(p_indices, q_indices)
        p, q = self.as_rational(p_indices, q_indices)
        gcd = math.gcd(p, q)
        reduced_q = mp.fabs(q // gcd)
        return -(1 + mp.log(mp.fabs(L - (p / q)), reduced_q))
