from __future__ import annotations

import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix
from ramanujantools.cmf import CMF

from functools import lru_cache

theta = sp.symbols("theta")


class MeijerG(CMF):
    r"""
    Represents the pFq CMF, derived from the differentiation property of generalized hypergeometric functions:
    https://en.wikipedia.org/wiki/Meijer_G-function
    """

    def __init__(self, m: int, n: int, p: int, q: int, z: sp.Expr = z):
        self.m = m
        self.n = n
        self.p = p
        self.q = q
        self.z = z
        matrices, negative_matrices = MeijerG.construct_matrices(m, n, p, q, z)
        super().__init__(
            matrices=matrices,
            _negative_matrices_cache=negative_matrices,
            validate=False,
        )

    @staticmethod
    def a_symbols(p) -> list[sp.Symbol]:
        return sp.symbols(f"a:{p}")

    @staticmethod
    def b_symbols(q) -> list[sp.Symbol]:
        return sp.symbols(f"b:{q}")

    @staticmethod
    def differential_equation(m, n, p, q, z) -> sp.Poly:
        coeff = (-sp.Integer(1)) ** (p - m - n)
        return sp.Poly(
            sp.expand(
                coeff * z * sp.prod(theta - a_i + 1 for a_i in MeijerG.a_symbols(p))
                - sp.prod(theta - b_j for b_j in MeijerG.b_symbols(q))
            ),
            theta,
        )

    @lru_cache
    @staticmethod
    def construct_matrices(
        m: int, n: int, p: int, q: int, z: sp.Expr
    ) -> tuple[dict[sp.Expr, Matrix], dict[sp.Expr, Matrix]]:
        if not (p > 0 and 0 <= n and n <= p) or not (q > 0 and 0 <= m and m <= q):
            raise ValueError(
                "Meijer G must satisfy p > 0, 0 <= n <= p and q > 0, 0 <= m <= q"
            )
        diff_eq = MeijerG.differential_equation(m, n, p, q, z)
        diff_eq_normalized = sp.Poly(diff_eq / sp.LC(diff_eq), theta)
        theta_matrix = Matrix.companion(diff_eq_normalized)

        r = theta_matrix.rows
        eye = Matrix.eye(r)

        def sign(i, threshold):
            return -1 if i > (threshold + 1) else 1

        matrices = {}
        negative_matrices = {}
        for i, a_i in enumerate(MeijerG.a_symbols(p)):
            negative_matrices[a_i] = (theta_matrix + (1 - a_i) * eye) * sign(i, n)
            matrices[a_i] = (
                negative_matrices[a_i].subs({a_i: a_i + 1}).inverse().factor()
            )

        for i, b_i in enumerate(MeijerG.b_symbols(q)):
            b_i_matrix = (-theta_matrix + b_i * eye) * sign(i, m)
            matrices[b_i] = b_i_matrix

        # Combine all into one dictionary
        return matrices, negative_matrices
