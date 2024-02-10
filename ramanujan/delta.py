import math

import mpmath as mp


def delta(p, q, L):
    r"""
    Calculates the irrationality measure $\delta$, defined as:
    $|\frac{p}{q} - L| = \frac{1}{q}^{1+\delta}$.

    Assumes both p and q are integers.
    """
    gcd = math.gcd(p, q)
    reduced_q = mp.fabs(q // gcd)
    return -(1 + mp.log(mp.fabs(L - (p / q)), reduced_q))
