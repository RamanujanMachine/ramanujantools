from __future__ import annotations
from typing import Union, List

import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix, Limit


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
            recurrence_matrix = Matrix.companion(list(reversed(column)))
        else:
            recurrence_matrix = recurrence.as_companion(inflate_all=False)
        if not recurrence_matrix.free_symbols.issubset({n}):
            raise ValueError("LinearRecurrence only supports the usage of the symbol n")
        self.recurrence_matrix = recurrence_matrix

    @property
    def relation(self) -> List[sp.Expr]:
        relation = self.recurrence_matrix.col(-1)
        denominator = relation.denominator_lcm
        return [denominator] + [c * denominator for c in reversed(relation)]

    def __repr__(self) -> str:
        return f"LinearRecurrence({self.relation})"

    def limit(self, iterations: int) -> Limit:
        r"""
        Returns the Limit matrix of the recursion up to a certain depth
        """
        return self.recurrence_matrix.limit({n: 1}, iterations, {n: 1})
