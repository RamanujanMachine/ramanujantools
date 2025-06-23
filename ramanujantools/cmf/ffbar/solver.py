import sympy as sp
from sympy.abc import x, y, n

from typing import Collection

from ramanujantools import GenericPolynomial
from ramanujantools.pcf import PCF
from ramanujantools.cmf.ffbar import FFbar


def polynomial_coefficients(poly, variables: Collection[sp.Symbol] = (x, y)) -> list:
    r"""
    Returns the coefficients of all monomials of `poly` in `variables`.
    By default assumes variables are $(x, y)$.
    """
    return sp.Poly(poly, *variables).coeffs()


def solve(expressions: sp.Expr) -> list:
    r"""
    Returns a list of solutions that solve `expressions`.

    The equation system represented by `expressions` is [expr = 0 for expr in expressions].
    """
    return sp.solve(expressions, dict=True, manual=True)


def from_pcf(pcf: PCF) -> list[FFbar]:
    r"""
    Attempts to construct a 2d FFbar CMF,
    such that Mx is the matrix representation of `pcf`,
    and My is as generic as possible.
    """
    a_deg, b_deg = pcf.degree()
    deg = max(a_deg, (b_deg + 1) // 2)
    f, _ = GenericPolynomial.of_combined_degree(deg=deg, var_name="c", variables=[x, y])
    fbar, _ = GenericPolynomial.of_combined_degree(
        deg=deg, var_name="d", variables=[x, y]
    )

    equations = [
        *polynomial_coefficients(
            pcf.a_n.subs(n, x) - FFbar.A(f.subs(y, 1), fbar.subs(y, 1))
        ),
        *polynomial_coefficients(
            pcf.b_n.subs(n, x) - FFbar.B(f.subs(y, 1), fbar.subs(y, 1))
        ),
    ]

    for solution in solve(equations):
        f = f.subs(solution).simplify()
        fbar = fbar.subs(solution).simplify()
        return solve_ffbar(f, fbar)


def solve_ffbar(f, fbar) -> list[FFbar]:
    r"""
    Returns all pairs of subsets of the input $f(x, y), \bar{f}(x, y)$ such that
    each one is a valid solution of the linear condition and the quadratic condition.
    """
    equations = [
        *polynomial_coefficients(FFbar.linear_condition(f, fbar)),
        *polynomial_coefficients(FFbar.quadratic_condition(f, fbar)),
    ]
    solutions = solve(equations)
    return [
        FFbar(f.subs(solution).simplify(), fbar.subs(solution).simplify())
        for solution in solutions
    ]
