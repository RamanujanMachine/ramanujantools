from __future__ import annotations

import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix
from ramanujantools.cmf import CMF

from functools import lru_cache


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

    @lru_cache
    @staticmethod
    def construct_matrices(
        m: int, n: int, p: int, q: int, z: sp.Expr
    ) -> tuple[dict[sp.Expr, Matrix], dict[sp.Expr, Matrix]]:
        if not (p > 0 and 0 <= n and n <= p) or not (q > 0 and 0 <= m and m <= q):
            raise ValueError(
                "Meijer G must satisfy p > 0, 0 <= n <= p and q > 0, 0 <= m <= q"
            )
        X = sp.symbols("X")
        a = sp.symbols(f"a:{p}")
        b = sp.symbols(f"b:{q}")

        # Construct the polynomial expression from the differential equation
        diff_eq_a = sp.prod([X - a_i + 1 for a_i in a])
        diff_eq_b = sp.prod([X - b_j for b_j in b])
        diff_eq = (-sp.Integer(1)) ** (p - m - n) * z * diff_eq_a - diff_eq_b
        diff_eq_coeffs = sp.Poly(diff_eq, X).all_coeffs()

        # Compute the recurrence coefficients T0, T1, ..., Tr
        r = len(diff_eq_coeffs) - 1
        eye = Matrix.eye(r)

        # Construct the Mtheta matrix (r x r with T as last column)
        T = [-diff_eq_coeffs[-(i + 1)] / diff_eq_coeffs[0] for i in range(r)]
        Mtheta = Matrix.companion_form(T)

        matrices = {}
        negative_matrices = {}
        # Build a_i matrices (bidiagonal form)
        for i, a_i in enumerate(a):
            a_i_matrix = Mtheta + (1 - a_i) * eye
            if i + 1 > n:
                a_i_matrix = -a_i_matrix
            negative_matrices[a_i] = a_i_matrix
            matrices[a_i] = a_i_matrix.subs({a_i: a_i + 1}).inv()

        # Build b_i matrices (bidiagonal form)
        for i, b_i in enumerate(b):
            b_i_matrix = -Mtheta + b_i * eye
            if i + 1 > m:
                b_i_matrix = -b_i_matrix
            matrices[b_i] = b_i_matrix

        # Combine all into one dictionary
        return matrices, negative_matrices
