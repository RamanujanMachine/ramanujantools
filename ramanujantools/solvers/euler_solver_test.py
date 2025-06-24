import pytest

import sympy
from sympy import Poly
from sympy.abc import x

from ramanujantools.solvers import EulerSolver, EulerSolution

"""
Looking for Euler solutions of the form
        $$b(x) = -h_1(x) h_1(x)$$
        $$f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$$

Either there is a trivial solution (f=1), general solution (general f), or no solution.
Right now, the EulerSolver only works when all the polynomials are over Z.
"""

sqrt5 = sympy.sqrt(5)


@pytest.mark.parametrize(
    "h1,h2",
    [
        # degree(h_1) = degree(h_2) = degree(a):=degree(h_1+h_2)
        (1, 1),
        (x**3, x**3),
        (x * x * (x + 5), x * (x - 3) * (x - 10)),
        # degree(h_1) = degree(h_2) > degree(a):=degree(h_1+h_2)
        (x**3, -(x**3)),
        (-(x**3), x**3),
        ((x + 1) * (x + 2), -(x + 1) * (x + 4)),
        # degree(h_1) < degree(h_2)
        (1, x),
        (1, x**2),
        (x + 1, x**2),
        ((x + 1), (x - 1) * (x - 2)),
        # degree(h_1) > degree(h_2)
        (x, 1),
        (x**2, 1),
        (x**2, x + 1),
        ((x + 1) * (x - 5) * (x + 9), (x - 1) * (x - 2)),
        # some non rational algebraic examples:
        ((1 + sqrt5) / 2, (1 - sqrt5) / 2),
        ((1 + sqrt5) / 2, x * (1 - sqrt5) / 2),
        (-x * (1 + sqrt5) / 2, x * (1 - sqrt5) / 2),
        (x**2 * (1 + sqrt5), x**2 * (1 - sqrt5)),
    ],
)
def test_trivial_euler_family(h1: Poly, h2: Poly):
    h2 = Poly(h2, x)  # needed for the subs(x, x+1) called below
    original_solution = EulerSolution(h1, h2, h1 + h2.subs(x, x + 1), -h1 * h2, 1)
    solutions = EulerSolver.solve_for(original_solution.a, original_solution.b)
    assert original_solution in solutions


@pytest.mark.parametrize(
    "h1,h2,f",
    [
        (x**4, x**4, 2 * x + 1),
        (x**5, x**5, 3 * x**2 + 3 * x + 1),
        (x**5, x**5, x**2 + x + 1),
        (x**5, x**5, 5 * x**4 + 10 * x**3 + 19 * x**2 + 14 * x + 4),
        (x**7, x**7, 2 * x**2 + 2 * x + 1),
    ],
)
def test_non_trivial_euler_family(h1: Poly, h2: Poly, f: Poly):
    h1 = Poly(h1, x)
    h2 = Poly(h2, x)
    b = Poly(-h1 * h2, x)

    f = Poly(f, x)
    g = Poly(f.subs(x, x - 1) * h1 + f.subs(x, x + 1) * h2.subs(x, x + 1), x)
    a, r = sympy.div(g, f)
    assert r == 0  # make sure that this is a valid solution

    original_solution = EulerSolution(h1, h2, a, b, f)
    solutions = EulerSolver.solve_for(original_solution.a, original_solution.b)
    assert original_solution in solutions


@pytest.mark.parametrize("a,b", [(3 * x + 1, x), (2 * x + 3, x), (x**2 + x**3, x**4)])
def test_no_euler_solution(a: Poly, b: Poly):
    assert len(EulerSolver.solve_for(a, b)) == 0


def test_solution_equality():
    # Two Euler solutions should be the same if they have the same h_1,h_2,a,b , and their f polynomials are
    # the same up to a scalar multiplication
    x_poly = Poly(x)
    sol1 = EulerSolution(h_1=x_poly, h_2=x_poly, a=x_poly, b=x_poly, f=x_poly)
    sol2 = EulerSolution(h_1=x_poly, h_2=x_poly, a=x_poly, b=x_poly, f=x_poly)
    assert sol1 == sol2

    sol1 = EulerSolution(h_1=x_poly, h_2=x_poly, a=x_poly, b=x_poly, f=-3 * x_poly)
    sol2 = EulerSolution(h_1=x_poly, h_2=x_poly, a=x_poly, b=x_poly, f=8 * x_poly)
    assert sol1 == sol2

    sol1 = EulerSolution(h_1=x_poly, h_2=x_poly, a=x_poly, b=x_poly, f=x_poly)
    sol2 = EulerSolution(h_1=x_poly, h_2=x_poly, a=2 * x_poly, b=x_poly, f=x_poly)
    assert sol1 != sol2

    sol1 = EulerSolution(h_1=x_poly, h_2=x_poly, a=x_poly, b=x_poly, f=2 * x_poly)
    sol2 = EulerSolution(h_1=x_poly, h_2=x_poly, a=2 * x_poly, b=x_poly, f=x_poly)
    assert sol1 != sol2


def test_non_unique_solution():
    # Tests a case where f is generic, and doesn't have any restrictions
    f_0 = sympy.Symbol("f_0")
    original_solution = EulerSolution(
        h_1=Poly(1, x), h_2=Poly(1, x), a=Poly(2, x), b=Poly(-1, x), f=Poly(x + f_0)
    )
    solution = EulerSolver.solve_for_decomposition_with_degree(
        a=original_solution.a,
        h_1=original_solution.h_1,
        h_2=original_solution.h_2,
        d_f=1,
    )
    assert original_solution == solution
