import sympy as sp
from sympy.abc import n

from ramanujan.pcf import PCF


def pfq(a, b, z):
    return sp.hyperexpand(sp.hyper(a, b, z))


def hypergeometric_limit(pcf: PCF):
    supported_degrees = {(1, 1): hyp_1f1_limit, (1, 2): hyp_2f1_limit}
    if pcf.degree() not in supported_degrees:
        raise ValueError(
            "Attempted to evaluate hypergeometric limit of {} of degree {}. Supported degrees are {}".format(
                pcf, pcf.degree(), supported_degrees.keys()
            )
        )
    return supported_degrees[pcf.degree](pcf)


def validate_degree(pcf: PCF, degree):
    if pcf.degree() != degree:
        raise ValueError(
            "Attempted to evaluate {}f{} limit of {} of degree {}. Only degree {} allowed.".format(
                degree[1], degree[0], pcf, pcf.degree(), degree
            )
        )


def hyp_1f1_limit(pcf: PCF):
    validate_degree(pcf, (1, 1))
    [a1, a0] = sp.Poly(pcf.a_n, n).all_coeffs()
    [b1, b0] = sp.Poly(pcf.b_n, n).all_coeffs()

    alpha = b0 / b1
    beta = (a0 * a1 + b1) / a1**2
    z = b1 / a1**2

    return sp.simplify(
        alpha * beta * (pfq([alpha], [beta], z) / pfq([alpha + 1], [beta + 1], z))
    )


def hyp_2f1_limit(pcf: PCF):
    validate_degree(pcf, (1, 2))
    [e, d] = sp.Poly(pcf.a_n, n).all_coeffs()
    [c, b, a] = sp.Poly(pcf.b_n, n).all_coeffs()
    delta = e**2 + 4 * c
    # assert delta > 0, "Delta is less than 0!"
    sqrt_delta = sp.sign(e) * sp.root(delta, 2)
    alpha, beta = sum(
        [
            ([sp.simplify(root)] * multiplicity)
            for root, multiplicity in sp.roots(c * n**2 - b * n + a, n).items()
        ],
        [],
    )
    z = sp.simplify((1 - e / sqrt_delta) / 2)
    gamma = sp.simplify(((b + c) * z) / c + d / sqrt_delta)
    return (
        sqrt_delta
        * gamma
        * (pfq([alpha, beta], [gamma], z) / pfq([alpha + 1, beta + 1], [gamma + 1], z))
    )
