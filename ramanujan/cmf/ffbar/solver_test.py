import sympy as sp
from sympy.abc import x, y

from ramanujan.cmf.ffbar import (
    linear_condition,
    solve_linear,
    quadratic_condition,
    solve_quadratic,
    solve,
)

c0 = sp.Symbol("c0")
c1 = sp.Symbol("c1")
c2 = sp.Symbol("c2")
c3 = sp.Symbol("c3")
c4 = sp.Symbol("c4")
c5 = sp.Symbol("c5")


def test_alone():
    f = c0 + c1 * x + c2 * y
    fbar = c3 + c4 * x + c5 * y
    linear_solutions = solve_linear(f, fbar)
    print(linear_solutions)
    for linear_solution in linear_solutions:
        print(linear_solution)
        _f = f.subs(linear_solution).simplify()
        _fbar = fbar.subs(linear_solution).simplify()
        quadratic_solutions = solve_quadratic(_f, _fbar)
        for quadratic_solution in quadratic_solutions:
            print(quadratic_solution)
            __f = _f.subs(quadratic_solution).simplify()
            __fbar = _fbar.subs(quadratic_solution).simplify()
            assert linear_condition(__f, __fbar) == 0
            assert quadratic_condition(__f, __fbar) == 0
            print(__f, ", ", __fbar)
    assert False


def test_together():
    f = c0 + c1 * x + c2 * y
    fbar = c3 + c4 * x + c5 * y
    solutions = solve([linear_condition(f, fbar), quadratic_condition(f, fbar)])
    for solution in solutions:
        print(f.subs(solution).simplify(), fbar.subs(solution).simplify())
    assert False
