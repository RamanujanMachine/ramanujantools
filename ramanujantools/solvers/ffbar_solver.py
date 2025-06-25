import sympy as sp
from sympy.abc import x, y, n

from typing import Collection

from ramanujantools import GenericPolynomial
from ramanujantools.pcf import PCF
from ramanujantools.cmf import FFbar


class FFbarSolver:
    @staticmethod
    def _polynomial_coefficients(
        poly, variables: Collection[sp.Symbol] = (x, y)
    ) -> list:
        r"""
        Returns the coefficients of all monomials of `poly` in `variables`.
        By default assumes variables are $(x, y)$.
        """
        return sp.Poly(poly, *variables).coeffs()

    @staticmethod
    def _solve_equations(expressions: sp.Expr) -> list:
        r"""
        Returns a list of solutions that solve `expressions`.

        The equation system represented by `expressions` is [expr = 0 for expr in expressions].
        """
        return sp.solve(expressions, dict=True, manual=True)

    @staticmethod
    def from_pcf(pcf: PCF) -> list[FFbar]:
        r"""
        Attempts to construct a 2d FFbar CMF,
        such that Mx is the matrix representation of `pcf`,
        and My is as generic as possible.
        """
        a_deg, b_deg = pcf.degrees()
        deg = max(a_deg, (b_deg + 1) // 2)
        f, _ = GenericPolynomial.of_combined_degree(
            deg=deg, var_name="c", variables=[x, y]
        )
        fbar, _ = GenericPolynomial.of_combined_degree(
            deg=deg, var_name="d", variables=[x, y]
        )

        equations = [
            *FFbarSolver._polynomial_coefficients(
                pcf.a_n.subs(n, x) - FFbar.A(f.subs(y, 1), fbar.subs(y, 1))
            ),
            *FFbarSolver._polynomial_coefficients(
                pcf.b_n.subs(n, x) - FFbar.B(f.subs(y, 1), fbar.subs(y, 1))
            ),
        ]

        for solution in FFbarSolver._solve_equations(equations):
            f = f.subs(solution).simplify()
            fbar = fbar.subs(solution).simplify()
            return FFbarSolver.solve_ffbar(f, fbar)

    @staticmethod
    def solve_ffbar(f: sp.Expr, fbar: sp.Expr) -> list[FFbar]:
        r"""
        Returns all pairs of subsets of the input $f(x, y), \bar{f}(x, y)$ such that
        each one is a valid solution of the linear condition and the quadratic condition.
        """
        equations = [
            *FFbarSolver._polynomial_coefficients(FFbar.linear_condition(f, fbar)),
            *FFbarSolver._polynomial_coefficients(FFbar.quadratic_condition(f, fbar)),
        ]
        solutions = FFbarSolver._solve_equations(equations)
        return [
            FFbar(f.subs(solution).simplify(), fbar.subs(solution).simplify())
            for solution in solutions
        ]
