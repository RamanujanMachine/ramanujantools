import sympy
from sympy import Poly, Symbol
from sympy.abc import x

from ramanujantools import Matrix, GenericPolynomial, simplify


class CoboundarySolver:
    r"""
    Used to find coboundary relations, namely given to parametrized matrices $m_1(x), m_2(x)$,
    is there a solution $m(x)$ to
    $m1(x) * m(x+1) = m(x) * m2(x)$.
    """

    @staticmethod
    def solve_polynomial_matrix(
        matrix: Matrix, symbol: Symbol, variables: list[Symbol]
    ) -> dict:
        r"""
        Given a polynomial matrix in symbol, such that the coefficients are over 'variables',
        try to find an assignment for this variables so that matrix(assignment) == 0.
        Return this assignment (possibly empty if there is no such assignment, or if the matrix is already zero).

        Note: Might not work if there are more variables than the given list in the parameters.
        """
        # TODO 1 : move this method to a more general location, so other places can use it, also
        # TODO 2 : consider polynomial matrices in several variables as well
        equations = (
            Poly(matrix[0, 0], symbol).all_coeffs()
            + Poly(matrix[1, 0], symbol).all_coeffs()
            + Poly(matrix[0, 1], symbol).all_coeffs()
            + Poly(matrix[1, 1], symbol).all_coeffs()
        )
        return sympy.solve(equations, variables)

    @staticmethod
    def find_coboundary(
        m1: Matrix, m2: Matrix, max_deg: int, symbol: Symbol = x
    ) -> tuple[Matrix, list[Symbol]] | None:
        r"""
        Given two parametrized matrices $m_1(s), m_2(s)$ over the given symbol,
        look for a parametrized polynomial matrix m(s) such that
                $ m_1(s) * m(s+1) = m(s) * m_2(s) $.
        The max degree for the polynomials in m(s) is the given max_deg.
        If there is such a matrix, then returns a tuple of the matrix and the free variables,
        otherwise, returns None.

        Note 1: A solution to the equation above is a vector space, so theoretically it should always
        contain the m(x)=0 solution, but as preparation for a more general solver, I add the None possibility.

        Note 2: Will probably not work if $m_1$ or $m_2$ depend on a variable different from the given symbol.
        """

        f11, f11_ = GenericPolynomial.of_degree(max_deg, "f11_", symbol)
        f12, f12_ = GenericPolynomial.of_degree(max_deg, "f12_", symbol)
        f21, f21_ = GenericPolynomial.of_degree(max_deg, "f21_", symbol)
        f22, f22_ = GenericPolynomial.of_degree(max_deg, "f22_", symbol)
        all_vars = f11_ + f12_ + f21_ + f22_

        # use '.expr' instead of just the polynomials, because calling subs(x,x+1) on polynomials in sympy
        # changes their "base" symbol to x+1 and then raises errors when trying to add them to polynomials
        # with base variable x.
        generic_m = Matrix([[f11.expr, f12.expr], [f21.expr, f22.expr]])

        equations = simplify(m1 * generic_m({symbol: symbol + 1}) - generic_m * m2)
        assignment = CoboundarySolver.solve_polynomial_matrix(
            equations, symbol, all_vars
        )

        if len(assignment) == 0:
            return None

        m = generic_m.subs(assignment)
        vars_left = [v for v in all_vars if v not in assignment.keys()]

        return m, vars_left

    @staticmethod
    def check_unique_solution(
        matrix: Matrix, variables: list[Symbol]
    ) -> tuple[Matrix, list[Symbol]]:
        r"""
        Assuming matrix is linearly dependent on the given variables, in case there is only 1 variable v in vars,
        check if matrix = v*matrix' and if so return (matrix', []).
        Otherwise, return (matrix,vars).
        """
        if len(variables) == 1:
            v = variables[0]
            if matrix.subs({v: 0}) == Matrix(
                [[0, 0], [0, 0]]
            ):  # the matrix m is linear in the variables
                return matrix.subs({v: 1}), []

        return matrix, variables
