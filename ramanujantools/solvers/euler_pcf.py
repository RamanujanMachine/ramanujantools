from dataclasses import dataclass
import math
import sympy as sp
from sympy.abc import x

from ramanujantools import GenericPolynomial


r"""
We say that a polynomial continued fraction (PCF) with functions $a(x), b(x)$ is in Euler's form,
if there are polynomial $h_1(x), h_2(x), f(x)$ such that

        $$b(x) = -h_1(x) h_1(x)$$
        $$f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$$

In case $f(x)=1$, we say that it has the trivial Euler form.

If a PCF has such a form, then its limit is
$ K_1^\infty \frac{b(n)}{a(n)} = h_2(0)*f_1(0)/f_1(0)  ( S^{-1} - 1 )^{-1} $
where
$ S = \sum_{k=0}^\infty \frac{ f(0)f(1) }{ f(k)f(k+1) } \prod_{i=1}^k \frac{ h_1(i) }{ h_2(i+1} } $
"""


def multi_subsets(elem_to_count: dict):
    """
    A generator returning all the decompositions of a multiset to two multisets.
    elem_to_count needs to be a dictionary from the distinct elements in the multisets to the number of times
    they appear there.

    For example, the multiset {{x, x, x, y, y, z}}, should be represented by the dictionary {x:3, y:2, z:1}
    """

    def _multi_subsets(inner_elem_to_count: dict, keys, current1, current2):
        if len(keys) == 0:
            yield current1, current2
            return

        key = keys[0]
        total = inner_elem_to_count[key]
        for amount in range(total + 1):
            yield from _multi_subsets(
                inner_elem_to_count,
                keys[1:],
                {**current1, key: amount},
                {**current2, key: total - amount},
            )

    yield from _multi_subsets(elem_to_count, list(elem_to_count.keys()), {}, {})


@dataclass
class EulerSolution:
    r"""
    A dataclass representing a polynomial continued fraction of the form :

        $b(x) = -h_1(x) h_1(x)$
        $f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$
    """

    h_1: sp.Poly
    h_2: sp.Poly
    a: sp.Poly
    b: sp.Poly
    f: sp.Poly

    def __post_init__(self):
        # make sure that all the parameters are actually polynomials
        self.h_1 = sp.Poly(self.h_1, x)
        self.h_2 = sp.Poly(self.h_2, x)
        self.a = sp.Poly(self.a, x)
        self.b = sp.Poly(self.b, x)
        self.f = sp.Poly(self.f, x)

    def __hash__(self):
        return hash(
            (
                tuple(self.h_1.all_coeffs()),
                tuple(self.h_2.all_coeffs()),
                tuple(self.a.all_coeffs()),
                tuple(self.b.all_coeffs()),
                tuple(self.f.all_coeffs()),
            )
        )

    @staticmethod
    def _poly_eq(poly1: sp.Poly, poly2: sp.Poly):
        """
        Use this equality in case the polynomials have non integer algebraic coefficients
        """
        coeffs1 = [sp.simplify(c) for c in poly1.all_coeffs()]
        coeffs2 = [sp.simplify(c) for c in poly2.all_coeffs()]
        if len(coeffs1) != len(coeffs2):
            return False
        return all(c1 == c2 for c1, c2 in zip(coeffs1, coeffs2))

    def __eq__(self, other):
        if not isinstance(other, EulerSolution):
            return False
        # f is determined up to scalar multiplication
        f_coef = self.f.all_coeffs()
        g_coef = other.f.all_coeffs()
        if len(f_coef) != len(g_coef):
            return False
        lead_f = f_coef[0]
        lead_g = g_coef[0]
        # check if f and g are the same up to scalar
        if any(
            lead_f * g_elem - lead_g * f_elem for f_elem, g_elem in zip(f_coef, g_coef)
        ):
            return False
        return (
            EulerSolution._poly_eq(self.h_1, other.h_1)
            and EulerSolution._poly_eq(self.h_2, other.h_2)
            and EulerSolution._poly_eq(self.a, other.a)
            and EulerSolution._poly_eq(self.b, other.b)
        )


class Coefficients:
    r"""
    Represents the coefficients of a polynomial, where at index i there is the coefficient of $x^i$.
    For all other indices, the returned coefficient value will be zero.
    """

    def __init__(self, p: sp.Poly):
        self.coef = list(reversed(p.all_coeffs()))

    def __repr__(self):
        return repr(self.coef)

    def __getitem__(self, index):
        if 0 <= index < len(self.coef):
            return self.coef[index]
        return 0


class EulerSolver:
    r"""
    A solver for the problem of given two polynomials $a(x), b(x)$, find polynomials $f(x), h_1(x), h_2(x)$
    such that:

        $b(x) = -h_1(x) h_1(x)$
        $f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$

    All the functions are static, in order to prevent using global functions.
    """

    @staticmethod
    def solve_for(a: sp.Poly, b: sp.Poly) -> list[EulerSolution]:
        r"""
        Given two polynomials, $a(x),b(x)$, searches for all solutions to
                $b(x)=-h_1(x)h_2(x)$,
                $f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$
        works under the assumption that $a(x),b(x)$ are integeral, and so are $h_1(x),h_2(x)$ in the solutions.
        Returns a list of all possible solutions.
        """
        # Make sure that the parameters are actually polynomials
        a = sp.Poly(a, x)
        b = sp.Poly(b, x)

        roots = b.all_roots()
        if len(roots) < b.degree():
            raise ValueError(f"Could not find all the roots for {b.expr}")

        # count multiplicity of roots
        roots_dict = {}
        for root in roots:
            if root in roots_dict:
                roots_dict[root] += 1
            else:
                roots_dict[root] = 1

        leading_coefficient = b.LC()

        return EulerSolver.solve_for_monic_decomposition(
            a, roots_dict, leading_coefficient
        )

    @staticmethod
    def solve_for_monic_decomposition(
        a: sp.Poly, b_roots: dict[int, int], leading_coefficient_b: int
    ) -> list[EulerSolution]:
        r"""
        Given $b(x)=leading_coefficient * \prod (x-root)$, finds all the solutions to
                $b(x)=-h_1(x)*h_2(x)$
                $f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)$
        and returns a list of these solutions
        """

        solutions = []
        for s1, s2 in multi_subsets(b_roots):
            # Each s_i is a dict {root: multiplicity}.
            # s1 and s2 are the decomposition of the roots of b(x) to two polynomials.
            # In particular, len(s1), len(s2) is going to be their degrees.

            deg_a = sp.degree(a)
            deg_h1 = sum(s1.values())
            deg_h2 = sum(s2.values())
            d = max(deg_a, deg_h1, deg_h2)
            if (deg_a == d) + (deg_h1 == d) + (deg_h2 == d) < 2:
                # At least two of the degrees should be maximal, otherwise there is no solution
                continue

            monic_poly_1 = sp.Poly(
                math.prod((x - root) ** power for root, power in s1.items()), x
            )
            monic_poly_2 = sp.Poly(
                math.prod((x - root) ** power for root, power in s2.items()), x
            )

            # we now have
            #       h_1 = c_1*monic_poly_1,
            #       h_2 = c_2*monic_poly_2
            # where
            #       c_1*c_2 = -leading_coefficient
            # If there is a solution, and d = max{deg(a), deg(h_1), deg(h_2)}, then
            #               coef_d(a)=coef_d(h_1)+coef_d(h_2)

            if deg_a < d:  # so that deg_a < d = deg_h1 == deg_h2
                # then c1 + c2 = 0, and c1*c2 = -c1*c1 = -leading_coefficient
                if leading_coefficient_b < 0:
                    continue  # TODO: add support to non integer solutions
                c = sp.sqrt(leading_coefficient_b)

                solutions += EulerSolver.solve_for_decomposition(
                    a, c * monic_poly_1, -c * monic_poly_2
                )
                solutions += EulerSolver.solve_for_decomposition(
                    a, -c * monic_poly_1, c * monic_poly_2
                )
                continue

            # now d=deg_a >= deg_h1, deg_h2
            leading_coefficient_a = a.LC()

            if deg_h1 < deg_h2:  # and automatically deg_a = deg_h2 = d
                c2 = leading_coefficient_a
                c1 = -leading_coefficient_b / c2
                solutions += EulerSolver.solve_for_decomposition(
                    a, c1 * monic_poly_1, c2 * monic_poly_2
                )
                continue

            if deg_h2 < deg_h1:  # and automatically deg_a = deg_h1 = d
                c1 = leading_coefficient_a
                c2 = -leading_coefficient_b / c1
                solutions += EulerSolver.solve_for_decomposition(
                    a, c1 * monic_poly_1, c2 * monic_poly_2
                )
                continue

            # Left with the case of deg_a = deg_h_1 = deg_h_2 = d, so that
            # c1 + c2 = leading_coefficient_a
            # c1 * c2 = - leading_coefficient_b
            # so that they are solutions to
            # (c - c1)*(c - c2) = c^2 - leading_coefficient_a * c - leading_coefficient_b
            disc = sp.sqrt(leading_coefficient_a**2 + 4 * leading_coefficient_b)
            c1 = (leading_coefficient_a + disc) / 2
            c2 = (leading_coefficient_a - disc) / 2

            solutions += EulerSolver.solve_for_decomposition(
                a, c1 * monic_poly_1, c2 * monic_poly_2
            )
            if c1 != c2:
                solutions += EulerSolver.solve_for_decomposition(
                    a, c2 * monic_poly_1, c1 * monic_poly_2
                )

        return solutions

    @staticmethod
    def solve_for_decomposition(
        a: sp.Poly, h_1: sp.Poly, h_2: sp.Poly
    ) -> list[EulerSolution]:
        """
        Tries to find a polynomial f solving the equation
                f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)
        Returns a list of such solutions, which can be empty if non exist.
        """
        degrees = EulerSolver.find_possible_degrees(a, h_1, h_2)
        if len(degrees) == 0:
            return []

        solutions = []
        for d in degrees:
            solution = EulerSolver.solve_for_decomposition_with_degree(a, h_1, h_2, d)
            if solution:
                solutions.append(solution)

        return solutions

    @staticmethod
    def find_possible_degrees(a: sp.Poly, h_1: sp.Poly, h_2: sp.Poly) -> list[int]:
        """
        Looking for the at most two possible degrees of a polynomial f which solves
                f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)
        return these degrees in a list (which has size 0, 1, or 2).
        """

        d = max(sp.degree(a), sp.degree(h_1), sp.degree(h_2))

        # Create lists of the coefficients of size d+1, such that p[i] will be the i-th coefficient of the polynomial p.
        a_coef = Coefficients(a)
        h_1_coef = Coefficients(h_1)
        h_2_plus_coef = Coefficients(sp.Poly(h_2.subs(x, x + 1), x))

        # Condition 1: Leading coefficients are equal.
        if a_coef[d] != h_1_coef[d] + h_2_plus_coef[d]:
            return []

        # Condition 2: The second leading coefficients are equal.
        #              The degree d_f appears in this equation, so unless it is 0=0, we can find d_f.
        num = a_coef[d - 1] - h_1_coef[d - 1] - h_2_plus_coef[d - 1]
        denom = h_2_plus_coef[d] - h_1_coef[d]
        if denom != 0:
            deg = float(sp.simplify(num / denom))
            if deg.is_integer() and int(deg) >= 0:
                return [int(deg)]
            return []

        if num != 0:
            return []

        # Condition 3: The third leading coefficients are equal. In case the previous condition didn't provide
        #              the degree, then we get here a nontrivial quadratic equation.
        aa = -(h_1_coef[d] + h_2_plus_coef[d])
        bb = h_1_coef[d - 1] - h_2_plus_coef[d - 1]
        cc = a_coef[d - 2] - h_1_coef[d - 2] - h_2_plus_coef[d - 2]

        coef2 = aa * 0.5
        coef1 = bb - aa * 0.5
        coef0 = cc

        # the possible degrees are the roots for coef2 * x^2 + coef1 * x + coef0 = 0
        disc_squared = coef1**2 - 4 * coef2 * coef0
        if disc_squared < 0:
            return []
        disc = math.sqrt(coef1**2 - 4 * coef2 * coef0)
        roots = [
            sp.simplify((-coef1 + disc) / (2 * coef2)),
            sp.simplify((-coef1 - disc) / (2 * coef2)),
        ]

        return [int(root) for root in roots if float(root).is_integer() and root >= 0]

    @staticmethod
    def solve_for_decomposition_with_degree(
        a: sp.Poly, h_1: sp.Poly, h_2: sp.Poly, d_f: int
    ) -> EulerSolution | None:
        """
        Tries to find a polynomial f of degree d_f solving the equation
                f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)
        If there is such a polynomial returns the solution, otherwise returns None.
        """

        b = -h_1 * h_2
        h_2_plus = sp.Poly(h_2.subs(x, x + 1), x)

        if d_f == 0:
            if a - (h_1 + h_2_plus) == 0:  # The only solution for constant polynomials
                return EulerSolution(
                    h_1=sp.Poly(h_1, x), h_2=sp.Poly(h_2, x), a=a, b=b, f=sp.Poly(1, x)
                )
            return None

        # Create the polynomial f, and the recursion it needs to solve, namely
        #    f(x)a(x) = f(x-1)h_1(x) + f(x+1)h_2(x+1)
        f, f_ = GenericPolynomial.of_degree(deg=d_f, var_name="f_", s=x, monic=True)
        f_plus = sp.Poly(f.subs(x, x + 1), x)
        f_minus = sp.Poly(f.subs(x, x - 1), x)

        p = f * a - f_minus * h_1 - f_plus * h_2_plus

        system = [sp.simplify(coefficient) for coefficient in p.all_coeffs()]
        system = [equation for equation in system if equation != 0]
        if len(system) == 0:  # no conditions on f_
            return EulerSolution(
                h_1=sp.Poly(h_1, x), h_2=sp.Poly(h_2, x), a=a, b=b, f=sp.Poly(f, x)
            )

        assignment = sp.solve(p.all_coeffs(), f_)
        if len(assignment) == 0:
            return None

        # TODO: If there is a solution, does it have to be unique?
        f_solved = f.subs(assignment)
        return EulerSolution(
            h_1=sp.Poly(h_1, x), h_2=sp.Poly(h_2, x), a=a, b=b, f=sp.Poly(f_solved, x)
        )
