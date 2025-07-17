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
        matrices = MeijerG.construct_matrices(m, n, p, q, z)
        super().__init__(matrices=matrices, validate=False)

    @lru_cache
    @staticmethod
    def construct_matrices(m: int, n: int, p: int, q: int, z: sp.Expr):
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
        T = Matrix([-diff_eq_coeffs[-(i + 1)] / diff_eq_coeffs[0] for i in range(r)])
        eye = Matrix.eye(r)

        # Construct the Mtheta matrix (r x r with T as last column)
        Mtheta = Matrix.hstack(Matrix.zeros(r, r - 1), T)

        # Build a_i matrices (bidiagonal form)
        Mai_dict = {}
        for i, a_i in enumerate(a):
            Mai = Mtheta + Matrix.companion_form([0] * r) + (1 - a_i) * eye
            if i + 1 > n:
                Mai = -Mai
            Mai_dict[a_i] = Mai.subs({a_i: a_i + 1}).inv()

        # Build b_i matrices (bidiagonal form)
        Mbi_dict = {}
        for i, b_i in enumerate(b):
            Mbi = -(Mtheta + Matrix.companion_form([0] * r)) + b_i * eye
            if i + 1 > m:
                Mbi = -Mbi
            Mbi_dict[b_i] = Mbi

        # Combine all into one dictionary
        return {**Mai_dict, **Mbi_dict}
