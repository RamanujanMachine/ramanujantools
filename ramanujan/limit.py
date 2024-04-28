from __future__ import annotations
from typing import List

from mpmath import mp

from ramanujan import Matrix, zero, inf


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

    def __eq__(self, other: Limit) -> bool:
        """
        Returns true iff two limits converge to the same value.
        """
        p, q = self.as_rational()
        other_p, other_q = other.as_rational()
        return p * other_q == q * other_p

    def precision(self, base: int = 10) -> int:
        """
        Returns the error in 'digits' for the PCF convergence.

        Args:
            base: The numerical base in which to return the precision (by default 10)
        """
        diff = abs(mp.mpq(*(self * zero())) - mp.mpq(*(self * inf())))
        return int(mp.floor(-mp.log(diff, 10)))

    def increase_precision(self) -> int:
        """
        Increases the global mpmath precision to the lever required to handle this limit.
        Returns the current precision after the increase.
        """
        requested_precision = self.precision() * 1.1  # Taking 10% digits buffer
        mp.dps = max(mp.dps, requested_precision)
        return mp.dps

    def as_rational(self) -> List:
        r"""
        Returns the limit as a rational number as a list [p, q],
        such that $m \cdot v = p/q$, where `m=self` and `v=vector`.
        """
        return list(self * zero())

    def as_float(self) -> mp.mpf:
        r"""
        Returns the limit as a floating point number f, such that $m \cdot v = f$, where `m=self` and `v=vector`.

        This function increases the global mpmath precision if needed.
        """
        self.increase_precision()
        p, q = self.as_rational()
        return mp.mpf(p) / mp.mpf(q)

    def as_rounded_number(self) -> str:
        """
        Same as `as_float`, but also rounds the result to the shortest number possible within the error range.

        This function increases the global mpmath precision if needed.
        """
        return most_round_in_range(self.as_float(), 10 ** -self.precision())
