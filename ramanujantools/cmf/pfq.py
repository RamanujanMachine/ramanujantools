from __future__ import annotations

import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix, Position
from ramanujantools.cmf import DFinite
from ramanujantools.cmf.d_finite import theta


class pFq(DFinite):
    r"""
    Represents the pFq CMF, derived from the differentiation property of generalized hypergeometric functions:
    https://en.wikipedia.org/wiki/Generalized_hypergeometric_function
    """

    def __init__(
        self,
        p: int,
        q: int,
        z: sp.Expr = z,
    ):
        r"""
        Constructs a pFq CMF.
        Args:
            p: The number of numerator parameters in the hypergeometric function
            q: The number of denominator parameters in the hypergeometric function
            z: If given, will attempt to construct the CMF for a specific z value.
        """
        self.p = p
        self.q = q
        self.z = sp.S(z)
        super().__init__(p, q, z)

    def __repr__(self) -> str:
        return f"pFq({self.p, self.q, self.z})"

    @staticmethod
    def x_axes(p: int) -> list[sp.Symbol]:
        return sp.symbols(f"x:{p}")

    @staticmethod
    def y_axes(q: int) -> list[sp.Symbol]:
        return sp.symbols(f"y:{q}")

    @classmethod
    def axes_and_signs(cls, p, q, z) -> dict[sp.Expr, bool]:
        x_axes = {x_i: True for x_i in pFq.x_axes(p)}
        y_axes = {y_i: False for y_i in pFq.y_axes(q)}
        return {**x_axes, **y_axes}

    @classmethod
    def differential_equation(cls, p, q, z) -> sp.Poly:
        return sp.Poly(
            sp.expand(
                theta * sp.prod(theta + y_i - 1 for y_i in pFq.y_axes(q))
                - z * sp.prod(theta + x_i for x_i in pFq.x_axes(p))
            ),
            theta,
        )

    @classmethod
    def construct_matrix(
        cls, theta_matrix: Matrix, axis: sp.Symbol, *args, **kwargs
    ) -> Matrix:
        eye = Matrix.eye(theta_matrix.rows)
        denom = axis - 1 if axis.name.startswith("y") else axis
        return theta_matrix / denom + eye

    def ascend(
        self, trajectory: Position, start: Position
    ) -> tuple[pFq, Position, Position]:
        r"""
        Returns a tuple of (CMF, trajectory, start), such that:
        1. The CMF is ascended, i.e, the CMF of _{p+1}F_{q+1}
        2. The start position contains two new more symbols
        3. The trajectory is padded with two zeros
        such that for any two choices of parameters $x_p, y_p$ such that $x_p - y_p \in \mathbb{Z}$,
        the ascended trajectory matrix contains all constants of the original,
        and the ascended delta is the same as the original (in a type-2 limit context).
        """
        ascended_cmf = pFq(
            self.p + 1,
            self.q + 1,
            self.z,
        )
        xp = sp.Symbol(f"x{self.p}")
        yq = sp.Symbol(f"y{self.q}")
        ascended_start = start + Position({xp: xp, yq: yq})
        ascended_trajectory = trajectory + Position({xp: 0, yq: 0})
        return (ascended_cmf, ascended_trajectory.sorted(), ascended_start.sorted())

    def subs(self, substitutions: Position) -> pFq:
        self._validate_axes_substitutions(substitutions)
        return pFq(
            self.p,
            self.q,
            self.z.subs(substitutions),
        )

    @staticmethod
    def predict_rank(p: int, q: int, z: sp.Expr):
        """
        Returns the rank of the CMF (i.e, the rank of its matrices).
        """
        N = max(p, q + 1)
        if z == 1 and p == q + 1:
            N -= 1
        return N

    @staticmethod
    def theta_derivative(expr: sp.Expr):
        r"""
        Returns the \theta derivative of an expression,
        which is defined as z * d/dz.
        """
        return z * sp.Derivative(expr, z).simplify()

    @staticmethod
    def state_vector(
        a_values: list[sp.Rational], b_values: list[sp.Rational], z_eval: sp.Expr = z
    ):
        r"""
        Returns the state vector of a pFq CMF in a specific point.
        The state vector is of length N, and the ith element is $\theta^i pFq(\bar{a}, \bar{b}, z)$,
        with a_values, b_values and z substituted in.
        """
        p = len(a_values)
        q = len(b_values)
        a_symbols = sp.symbols(f"a:{p}")
        b_symbols = sp.symbols(f"b:{q}")
        values = [sp.hyper(a_symbols, b_symbols, z)]
        for _ in range(1, pFq.predict_rank(p, q, z_eval)):
            values.append(pFq.theta_derivative(values[-1]))
        a_subs = Position.from_list(a_values, "a")
        b_subs = Position.from_list(b_values, "b")
        return sp.hyperexpand(
            Matrix(values).transpose().subs(a_subs | b_subs | {z: z_eval})
        )

    @staticmethod
    def contiguous_relation(point, anchor, z) -> Matrix:
        a_point, b_point = point
        a_anchor, b_anchor = anchor
        p = len(a_point)
        q = len(b_point)
        start = Position.from_list(a_anchor, "x") | Position.from_list(b_anchor, "y")
        end = Position.from_list(a_point, "x") | Position.from_list(b_point, "y")
        return pFq(p, q, z).work(start, end)

    @staticmethod
    def evaluate(
        a_values: list[sp.Rational], b_values: list[sp.Rational], z_eval: sp.Rational
    ) -> sp.Expr:
        """
        Evaluates symbolically the pFq function at a specific point.
        Levarages the pFq CMF to calculate the contiguous relations.
        Works by selecting a small anchor point for which we calculate the state vector
        using sympy, and then calculating the `work` matrix which represents the contiguous
        relations transformation up to our desired point.
        """
        a_values = [sp.S(value) for value in a_values]
        b_values = [sp.S(value) for value in b_values]
        a_anchor = [
            sp.sign(value) * (value - (value.floor() - 1)) for value in a_values
        ]
        b_anchor = [
            sp.sign(value) * (value - (value.floor() - 2)) for value in b_values
        ]
        vector = pFq.state_vector(a_anchor, b_anchor, z_eval)
        m = pFq.contiguous_relation((a_values, b_values), (a_anchor, b_anchor), z_eval)
        return (vector * m)[0]
