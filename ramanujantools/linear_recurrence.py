from __future__ import annotations
from typing import Union, List, Dict, Set
import itertools

import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix, Limit, GenericPolynomial


class LinearRecurrence:
    r"""
    Represents a linear recurrence of the form
    $a_0(n) p(n) = \sum_{i=1}^{N}a_i(n) p(n-i)$
    """

    def __init__(self, recurrence: Union[Matrix, List[sp.Expr]]):
        r"""
        Construct the recurrence.

        The recurrence argument can be one of two types:
            1. A list of the coefficients of the recurrence [a_0(n), ..., a_
        """
        if type(recurrence) is not Matrix:
            if len(recurrence) == 0:
                raise ValueError("Attempted to construct an empty recurrence!")
            denominator = sp.simplify(recurrence[0])
            column = [c / denominator for c in recurrence[1:]]
            recurrence_matrix = Matrix.companion_form(list(reversed(column)))
        else:
            recurrence_matrix = recurrence.as_companion(inflate_all=False)
        self.recurrence_matrix = recurrence_matrix

    def __eq__(self, other: Matrix) -> bool:
        return self.recurrence_matrix == other.recurrence_matrix

    def __repr__(self) -> str:
        return f"LinearRecurrence({self.relation()})"

    def subs(self, substitutions: Dict) -> LinearRecurrence:
        return LinearRecurrence([p.subs(substitutions) for p in self.relation()])

    def relation(self) -> List[sp.Expr]:
        relation = self.recurrence_matrix.col(-1)
        denominator = relation.denominator_lcm
        return [denominator] + [
            sp.simplify(c * denominator) for c in reversed(relation)
        ]

    def free_symbols(self) -> Set[sp.Symbol]:
        return set.union(*[p.free_symbols for p in self.relation()]) - {n}

    def limit(self, iterations: int, start=1) -> Limit:
        r"""
        Returns the Limit matrix of the recursion up to a certain depth
        """
        return self.recurrence_matrix.limit({n: 1}, iterations, {n: start})

    def compose(self, composition: Dict) -> LinearRecurrence:
        relation = self.relation()
        for index, amount in composition.items():
            shift = n + 1 - index
            modification = (
                [0] * shift
                + [-amount * self.relation()[0].subs({n: index - 1})]
                + [amount * c.subs({n: index - 1}) for c in self.relation()[1:]]
            )
            relation = [
                sum(d)
                for d in itertools.zip_longest(relation, modification, fillvalue=0)
            ]
        return LinearRecurrence(relation)

    @staticmethod
    def degree(p):
        return max(sp.Poly(p, n).degree(), 1)

    @staticmethod
    def generic_relation(degrees: List[int]) -> List[sp.Expr]:
        relation = []
        variable = "a"
        for degree in degrees:
            poly, _ = GenericPolynomial.of_degree(degree, variable, n)
            variable = chr(ord(variable) + 1)
            relation.append(poly.as_expr())
        return relation

    @staticmethod
    def all_divisors(p: sp.Poly) -> List[sp.Poly]:
        p = sp.Poly(p, n)
        content, factors_list = p.factor_list()
        factors = []
        for factor, order in factors_list:
            factors.append([factor**d for d in range(order + 1)])
        for root, order in content.factors().items():
            factors.append([root**d for d in range(order + 1)])
        combinations = itertools.product(*factors)
        divisors = []
        for combination in combinations:
            divisors.append(sp.prod(combination))
        return divisors

    def possible_decompositions(self) -> List[sp.Poly]:
        return LinearRecurrence.all_divisors(self.relation()[-1])

    def decompose_degree(self, degree: int) -> List:
        results = []
        relation = self.relation()
        composition, _ = GenericPolynomial.of_degree(degree, "C", n)
        composition = composition.as_expr()
        lead = relation[0]
        tailing = LinearRecurrence.generic_relation(
            [degree + LinearRecurrence.degree(p) for p in relation[1:-1]]
        )
        recurrence = LinearRecurrence([lead] + tailing)
        generic = recurrence.compose({n: composition})
        polynomials = [
            sp.Poly(p - q, n)
            for p, q in itertools.zip_longest(
                generic.relation()[1:], relation[1:], fillvalue=0
            )
        ]
        equations = sum([p.coeffs() for p in polynomials], [])
        solutions = sp.solve(equations)
        for solution in solutions:
            results.append((recurrence.subs(solution), {n: composition.subs(solution)}))
        return results

    def decompose(self, early_exit=True):
        """
        For now only attempting to reduce depth by 1
        """
        results = []
        relation = self.relation()
        max_composition_degree = min([LinearRecurrence.degree(p) for p in relation[1:]])
        if early_exit:
            for d in range(max_composition_degree):
                results = self.decompose_degree(d)
                if len(results) > 0:
                    return results
            return []
        else:
            return self.decompose_degree(max_composition_degree)
