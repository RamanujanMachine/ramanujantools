import sympy as sp
from sympy.abc import n

from ramanujan.pcf import PCF


def pfq(a, b, z):
    return sp.hyperexpand(sp.hyper(a, b, z))


def hypergeometric_limit(pcf: PCF):
    if pcf.degree() == (1, 1):
        return hyp_1f1_limit(pcf)


def hyp_1f1_limit(pcf: PCF):
    if pcf.degree() != (1, 1):
        raise ValueError(
            "Attempted to evaluate 1f1 limit of {} with degree {}. Only degree {} allowed.".format(
                pcf, pcf.degree(), (1, 1)
            )
        )
    [a1, a0] = sp.Poly(pcf.a_n, n).all_coeffs()
    [b1, b0] = sp.Poly(pcf.b_n, n).all_coeffs()

    alpha = b0 / b1
    beta = (a0 * a1 + b1) / a1**2
    z = b1 / a1**2

    return sp.simplify(
        alpha * beta * (pfq([alpha], [beta], z) / pfq([alpha + 1], [beta + 1], z))
    )
