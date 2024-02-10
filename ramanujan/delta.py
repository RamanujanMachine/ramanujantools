import math

import mpmath as mp


def delta(p, q, L):
    gcd = math.gcd(p, q)
    reduced_q = mp.fabs(q // gcd)
    return -(1 + mp.log(mp.fabs(L - (p / q)), reduced_q))
