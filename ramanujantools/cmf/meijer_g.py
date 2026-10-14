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

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.update({"m": self.m, "n": self.n, "p": self.p, "q": self.q, "z": self.z})
        return state

    def __setstate__(self, state):
        self.m = state["m"]
        self.n = state["n"]
        self.p = state["p"]
        self.q = state["q"]
        self.z = state["z"]
        del state["m"]
        del state["n"]
        del state["p"]
        del state["q"]
        del state["z"]
        super().__setstate__(state)

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
            sign = -1 if index >= n else 1
        else:
            sign = 1 if index >= m else -1
        multiplier = axis - 1 if is_a else axis
        return (theta_matrix - multiplier * eye) * sign

    @staticmethod
    def determinant(m: int, n: int, p: int, q: int, z: sp.Expr, axis: sp.Symbol):
        r"""
        Returns the determinant of an axis (basis) matrix in factored form, without computing ``det()``.

        Let $P(\theta)$ be the differential equation, $C$ the companion matrix of the monic $P$,
        $s=\pm1$ the axis sign of `construct_matrix` and $d=\deg P$.
        Substituting the contiguous relation of the axis into $P$ gives the characteristic polynomial of the shift:

        - b axes: $M_b = s (C - b I)$, so $\det M_b = (-s)^d \cdot \frac{P}{LC(P)}(\theta=b)$.
        - a axes: the matrix is the inverse of $s (C - (a-1) I)$ with $a \to a+1$, so
          $\det M_a = 1 / \left[(-s)^d \cdot \frac{P}{LC(P)}\Big|_{a \to a+1}(\theta=a)\right]$.
          The shift also applies to the leading coefficient (relevant when $z=\pm1$ cancels the top coefficient).

        :param axis: an $a_i$ or $b_j$ symbol of the CMF.
        :raises ValueError: if axis is not an a/b axis of the CMF.
        """
        name = axis.name
        if not (
            name[:1] in ("a", "b")
            and name[1:].isdigit()
            and int(name[1:]) < (p if name[0] == "a" else q)
        ):
            raise ValueError(f"not a Meijer-G axis: {axis}")
        index = int(name[1:])
        if name.startswith("a"):
            sign = -1 if index >= n else 1
        else:
            sign = 1 if index >= m else -1
        P = MeijerG.differential_equation(m, n, p, q, z)
        monic = P.as_expr() / P.LC()
        factor = (-sign) ** P.degree()
        if name.startswith("b"):
            return sp.factor(factor * monic.subs(theta, axis))
        lowered = monic.subs(axis, axis + 1).subs(theta, axis)
        return sp.factor(1 / (factor * lowered))
