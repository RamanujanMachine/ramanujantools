import sympy as sp
from sympy.abc import x, y

from ramanujan.cmf.ffbar import linear_condition, quadratic_condition


def solve_linear(f, fbar):
    return solve([linear_condition(f, fbar)])


def solve_quadratic(f, fbar):
    return solve([quadratic_condition(f, fbar)])


def solve(expressions):
    variables = set()
    for expression in expressions:
        variables.update(expression.free_symbols)
    variables -= {x, y}
    return sp.solve(expressions, variables, dict=True)
