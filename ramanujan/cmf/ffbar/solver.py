import sympy as sp
from sympy.abc import x, y

from ramanujan.cmf.ffbar import linear_condition, quadratic_condition


def polynomial_coefficients(poly, variables=(x, y)):
    """
    Returns the coefficients of all monomials of `poly` in `variables`.
    By default assumes variables are $(x, y)$.
    """
    return sp.Poly(poly, *variables).coeffs()


def solve(expressions):
    """
    Returns a list of solutions that solve `expressions`.

    The equation system represented by `expressions` is [expr = 0 for expr in expressions].
    """
    return sp.solve(expressions, dict=True, manual=True)


def solve_ffbar(f, fbar):
    """
    Returns all pairs of subsets of the input $f(x, y), \bar{f}(x, y)$ such that
    each one is a valid solution of the linear condition and the quadratic condition.
    """
    equations = [
        *polynomial_coefficients(linear_condition(f, fbar)),
        *polynomial_coefficients(quadratic_condition(f, fbar)),
    ]
    solutions = solve(equations)
    return [
        (f.subs(solution).simplify(), fbar.subs(solution).simplify())
        for solution in solutions
    ]
