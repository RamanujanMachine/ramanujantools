import sympy as sp
from sympy.abc import x, y

from ramanujan.cmf.ffbar import linear_condition, quadratic_condition


def polynomial_coefficients(poly):
    return sp.Poly(poly, x, y).coeffs()


def full_poly(letter: str, degree: int):
    poly = 0
    for x_deg in range(degree + 1):
        for y_deg in range(degree + 1 - x_deg):
            symbol = sp.Symbol(f"{letter}{x_deg}{y_deg}")
            poly += symbol * (x**x_deg) * (y**y_deg)
    return poly


def extract_variables(expressions):
    variables = set()
    for expression in expressions:
        variables.update(expression.free_symbols)
    variables -= {x, y}
    return variables


def rename_variables(expressions, letter: str):
    current_variables = list(extract_variables(expressions))
    substitutions = {}
    for i in range(len(current_variables)):
        substitutions.update({current_variables[i]: sp.Symbol(f"{letter}{i}")})
    return [expression.subs(substitutions).simplify() for expression in expressions]


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
