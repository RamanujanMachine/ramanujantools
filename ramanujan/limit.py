from typing import List

from mpmath import mp

from ramanujan import Matrix, zero, inf


class Limit(Matrix):
    r"""
    Represents a mathematical limit of a `walk` operation.
    """

    def as_rational(self) -> List:
        r"""
        Returns the limit as a rational number as a list [p, q],
        such that $m \cdot v = p/q$, where `m=self` and `v=vector`.
        """
        return list(self * zero())

    def as_float(self) -> mp.mpf:
        r"""
        Returns the limit as a floating point number f,
        such that $m \cdot v = f$, where `m=self` and `v=vector`.
        """
        p, q = self.as_rational()
        return mp.mpf(p) / mp.mpf(q)

    def precision(self, base: int = 10) -> int:
        """
        Returns the error in 'digits' for the PCF convergence.

        If `m` is not a result of a `PCF.walk` multiplication,
        or if the pcf does not converge this function is undefined.
        By default, `base` is 10 (for digits).
        """
        diff = abs(mp.mpq(*(self * zero())) - mp.mpq(*(self * inf())))
        return int(mp.floor(-mp.log(diff, 10)))
