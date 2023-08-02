import sympy
from sympy import Poly, Symbol
from sympy.abc import x
from typing import List, Optional, Tuple

from ramanujan.matrix import Matrix
from ramanujan.simplify_object import simplify


class CoboundarySolver:

    @staticmethod
    def generic_poly(deg: int, var_name: str, s: Symbol) -> Tuple[Poly, List[Symbol]]:
        r"""
        Creates and returns a generic polynomial of degree 'deg' with a list of its generic symbols
        """
        # TODO : move this method to a more general location, so other places can use it
        poly_vars = sympy.symbols(f'{var_name}:{deg + 1}')
        poly = Poly(sum(poly_vars[i] * s ** i for i in range(deg + 1)), s)
        return poly, poly_vars

    @staticmethod
    def find_coboundary(
            m1: Matrix, m2: Matrix, max_deg: int, symbol: Symbol = x) -> Optional[Tuple[Matrix, List[Symbol]]]:
        r"""
        Given two parametrized matrices m1(s), m2(s) over the given symbol,
        look for a parametrized polynomial matrix m(s) such that
                $ m1(s) * m(s+1) = m(s) * m2(s) $.
        The max degree for the polynomials in m(s) is the given max_deg.
        If there is such a matrix, then returns a tuple of the matrix and the free variables,
        otherwise, returns None.
        """

        f11, f11_ = CoboundarySolver.generic_poly(max_deg, 'f11_', symbol)
        f12, f12_ = CoboundarySolver.generic_poly(max_deg, 'f12_', symbol)
        f21, f21_ = CoboundarySolver.generic_poly(max_deg, 'f21_', symbol)
        f22, f22_ = CoboundarySolver.generic_poly(max_deg, 'f22_', symbol)
        all_vars = f11_ + f12_ + f21_ + f22_

        # use '.expr' instead of just the polynomials, because calling subs(x,x+1) on polynomials in sympy
        # changes their "base" symbol to x+1 and then raises errors when trying to add them to polynomials
        # with base variable x.
        generic_m = Matrix([[f11.expr, f12.expr], [f21.expr, f22.expr]])

        equations = simplify(m1 * generic_m({symbol: symbol+1}) - generic_m * m2)
        solution = sympy.solve(Poly(equations[0, 0], symbol).all_coeffs() +
                               Poly(equations[1, 0], symbol).all_coeffs() +
                               Poly(equations[0, 1], symbol).all_coeffs() +
                               Poly(equations[1, 1], symbol).all_coeffs(), all_vars)

        if len(solution)==0:
            return None

        m = generic_m.subs(solution)
        vars_left = [v for v in all_vars if v not in solution.keys()]

        # in case there is only 1 variable v left, check if m = v*m', and if so return m' instead.
        if len(vars_left) == 1:
            v = vars_left[0]
            if m.subs({v: 0}) == Matrix([[0, 0], [0, 0]]):   # the matrix m is linear in the variables
                m = m.subs({v: 1})
                vars_left = []

        return m, vars_left
