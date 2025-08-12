from __future__ import annotations

import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix
from ramanujantools.cmf import DFinite
from ramanujantools.cmf.d_finite import theta


class MeijerG(DFinite):
    r"""
    Represents the pFq CMF, derived from the differentiation property of generalized hypergeometric functions:
    https://en.wikipedia.org/wiki/Meijer_G-function
    """

    def __init__(self, m: int, n: int, p: int, q: int, z: sp.Expr = z):
        if not (p > 0 and 0 <= n and n <= p) or not (q > 0 and 0 <= m and m <= q):
            raise ValueError(
                "Meijer G must satisfy p > 0, 0 <= n <= p and q > 0, 0 <= m <= q"
            )
        self.m = m
        self.n = n
        self.p = p
        self.q = q
        self.z = z
        super().__init__(self.m, self.n, self.p, self.q, self.z)

    def __repr__(self) -> str:
        return f"MeijerG({self.m, self.n, self.p, self.q, self.z})"

    @staticmethod
    def a_axes(p) -> list[sp.Symbol]:
        return sp.symbols(f"a:{p}")

    @staticmethod
    def b_axes(q) -> list[sp.Symbol]:
        return sp.symbols(f"b:{q}")

    @classmethod
    def axes_and_signs(cls, m, n, p, q, z) -> dict[sp.Expr, bool]:
        a_axes = {a_i: False for a_i in MeijerG.a_axes(p)}
        b_axes = {b_i: True for b_i in MeijerG.b_axes(q)}
        return {**a_axes, **b_axes}

    @staticmethod
    def differential_equation(m, n, p, q, z) -> sp.Poly:
        coeff = (-sp.Integer(1)) ** (p - m - n)
        return sp.Poly(
            sp.expand(
                coeff * z * sp.prod(theta - a_i + 1 for a_i in MeijerG.a_axes(p))
                - sp.prod(theta - b_j for b_j in MeijerG.b_axes(q))
            ),
            theta,
        )

    @classmethod
    def construct_matrix(
        cls, theta_matrix: Matrix, axis: sp.Symbol, m: int, n: int, *args, **kwargs
    ) -> Matrix:
        eye = Matrix.eye(theta_matrix.rows)
        is_a = axis.name.startswith("a")
        index = int(axis.name[1:])
        if is_a:
            sign = -1 if index > n else 1
        else:
            sign = 1 if index > m else -1
        multiplier = axis - 1 if is_a else axis
        return (theta_matrix - multiplier * eye) * sign
