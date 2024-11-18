from __future__ import annotations
from typing import Union, List, Dict, Set
import copy
import itertools
from multimethod import multimethod

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
            1. A list of the coefficients of the recurrence [a_0(n), ..., a_N(n)]
        """
        if type(recurrence) is not Matrix:
            if len(recurrence) == 0:
                raise ValueError("Attempted to construct an empty recurrence!")
            gcd = sp.gcd(recurrence)
            self.relation = [sp.simplify(p / gcd) for p in recurrence]
        else:
            recurrence_matrix = recurrence.as_companion(inflate_all=False)
            col = recurrence_matrix.col(-1)
            lead = col.denominator_lcm
            coeffs = [sp.simplify(p * lead) for p in reversed(col)]
            self.relation = [lead] + coeffs

    @property
    def recurrence_matrix(self):
        denominator = sp.simplify(self.relation[0])
        column = [c / denominator for c in self.relation[1:]]
        return Matrix.companion_form(list(reversed(column)))

    def depth(self):
        return len(self.relation) - 1

    def degrees(self):
        return [LinearRecurrence.degree(p) for p in self.relation]

    def __eq__(self, other: Matrix) -> bool:
        return self.relation == other.relation

    def __repr__(self) -> str:
        return f"LinearRecurrence({self.relation})"

    def subs(self, substitutions: Dict) -> LinearRecurrence:
        return LinearRecurrence([p.subs(substitutions) for p in self.relation])

    def free_symbols(self) -> Set[sp.Symbol]:
        return set.union(*[p.free_symbols for p in self.relation]) - {n}

    def simplify(self) -> LinearRecurrence:
        return LinearRecurrence([sp.factor(p.simplify()) for p in self.relation])

    def limit(self, iterations: int, start=1) -> Limit:
        r"""
        Returns the Limit matrix of the recursion up to a certain depth
        """
        return self.recurrence_matrix.limit({n: 1}, iterations, {n: start})

    def compose(self, composition: sp.Expr) -> LinearRecurrence:
        relation = self.relation
        modification = (
            [0]
            + [-composition * self.relation[0].subs({n: n - 1})]
            + [composition * c.subs({n: n - 1}) for c in self.relation[1:]]
        )
        relation = [
            sum(d) for d in itertools.zip_longest(relation, modification, fillvalue=0)
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
        return LinearRecurrence.all_divisors(self.relation[-1])

    @multimethod
    def decompose(self, decomposition: sp.Poly):
        decomposed = [self.relation[0]]
        for index in range(1, len(self.relation) - 1):
            next = (-1 if index == 1 else 1) * decomposed[index - 1].subs({n: n - 1})
            decomposed.append(
                sp.simplify(self.relation[index] - next * decomposition.as_expr())
            )
        if decomposed[-1].subs({n: n - 1}) * decomposition == self.relation[-1]:
            return LinearRecurrence(decomposed)
        return None

    @multimethod
    def decompose(self):  # noqa: F811
        results = []
        for decomposition in self.possible_decompositions():
            recurrence = self.decompose(decomposition)
            if recurrence is not None:
                results.append((recurrence, decomposition))
        return results

    def inflate(self, p: sp.Expr) -> LinearRecurrence:
        p = sp.simplify(p.as_expr())
        current = p
        relation = copy.deepcopy(self.relation)
        for i in range(1, len(self.relation)):
            relation[i] *= current
            current *= p.subs({n: n - i})
        return LinearRecurrence(relation)

    def deflate(self, p: sp.Expr) -> LinearRecurrence:
        recurrence = self.inflate(1 / p)
        denominators = []
        for r in recurrence.relation:
            denominators.append(r.as_numer_denom()[1])
        lcm = sp.lcm(denominators)
        for i in range(len(recurrence.relation)):
            recurrence.relation[i] *= lcm
        return recurrence.simplify()
