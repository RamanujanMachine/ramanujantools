import sympy as sp
from sympy.abc import x, y

from ramanujan.cmf.ffbar import linear_condition, quadratic_condition


def polynomial_coefficients(poly):
    return sp.Poly(poly, x, y).coeffs()


def extract_variables(expressions):
    variables = set()
    for expression in expressions:
        variables.update(expression.free_symbols)
    variables -= {x, y}
    return variables


def solve(expressions):
    return sp.solve(expressions, dict=True, manual=True)


def solve_ffbar(f, fbar):
    equations = [
        *polynomial_coefficients(linear_condition(f, fbar)),
        *polynomial_coefficients(quadratic_condition(f, fbar)),
    ]
    solutions = solve(equations)
    return [
        (f.subs(solution).simplify(), fbar.subs(solution).simplify())
        for solution in solutions
    ]
