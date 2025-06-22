from __future__ import annotations

import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix, Position
from ramanujantools.cmf import CMF

from functools import lru_cache


theta = sp.symbols("theta")


class pFq(CMF):
    r"""
    Represents the pFq CMF, derived from the differentiation property of generalized hypergeometric functions:
    https://en.wikipedia.org/wiki/Generalized_hypergeometric_function

    """

    def __init__(
        self,
        p: int,
        q: int,
        z_eval: sp.Expr = z,
        theta_derivative: bool = True,
        negate_denominator_params: bool = True,
    ):
        r"""
        Constructs a pFq CMF.
        A few things to note when working with the pFq CMF construction:
        1. Construction is possible using either normal or theta derivatives
        2. While the construction is purely symbolic, however singularity is possible in some z values.
            To overcome this, we can select z during construction and construct specifically for it.
        3. The y matrices are inversed, as they represent a negative step in the matching y axis.
            There are two options to overcome this: inverse the matrices, or negate all y occurences.

        Args:
            p: The number of numerator parameters in the hypergeometric function
            q: The number of denominator parameters in the hypergeometric function
            z_eval: If given, will attempt to construct the CMF for a specific z value.
            theta_derivative: If set to False, will construt the CMF using normal derivatives.
                Otherwise, will use theta derivatives.
            negate_denominator_params: if set to True, will inverse all y matrices.
                Otherwise, will substitute y with -y.
        """
        matrices = pFq.construct_matrices(
            p, q, z_eval, theta_derivative, negate_denominator_params
        )
        self.p = p
        self.q = q
        self.z = sp.S(z_eval)
        self.theta_derivative = theta_derivative
        self.negate_denominator_params = negate_denominator_params
        super().__init__(matrices=matrices, validate=False)

    @lru_cache
    @staticmethod
    def construct_matrices(
        p: int,
        q: int,
        z_eval: sp.Expr = z,
        theta_derivative: bool = True,
        negate_denominator_params: bool = True,
    ):
        r"""
        Constructs the pFq CMF matrices.
        """
        x = sp.symbols(f"x:{p}")
        y = sp.symbols(f"y:{q}")

        def differential_equation(p, q) -> sp.Poly:
            return sp.Poly(
                sp.expand(
                    theta * sp.prod(theta + y[i] - 1 for i in range(q))
                    - z_eval * sp.prod(theta + x[i] for i in range(p))
                ),
                theta,
            )

        def core_matrix(p, q) -> Matrix:
            d_poly = differential_equation(p, q)
            d_poly_monic = sp.Poly(d_poly / sp.LC(d_poly), theta)
            return Matrix.companion(d_poly_monic)

        M = core_matrix(p, q)

        equation_size = M.rows

        if not theta_derivative:
            basis_transition_matrix = Matrix(
                equation_size,
                equation_size,
                lambda i, j: sp.functions.combinatorial.numbers.stirling(j, i)
                * (z_eval**i),
            )
            M = (
                basis_transition_matrix * M * (basis_transition_matrix.inverse())
            ).factor()

        if negate_denominator_params:
            y_matrices = {
                y[i]: Matrix(
                    M.subs({y[i]: y[i] + 1}) / y[i] + Matrix.eye(equation_size)
                )
                .inverse()
                .factor()
                for i in range(q)
            }
        else:
            M = M.subs({y[i]: -y[i] for i in range(q)})
            y_matrices = {
                y[i]: Matrix(-M / (y[i] + 1) + Matrix.eye(equation_size))
                for i in range(q)
            }

        matrices = {
            x[i]: Matrix(M / x[i] + Matrix.eye(equation_size)) for i in range(p)
        }
        matrices.update(y_matrices)
        return matrices

    def __repr__(self) -> str:
        return f"pFq({self.p, self.q, self.z, self.theta_derivative, self.negate_denominator_params})"

    def ascend(
        self, trajectory: Position, start: Position
    ) -> tuple[CMF, Position, Position]:
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
            self.theta_derivative,
            self.negate_denominator_params,
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
            self.theta_derivative,
            self.negate_denominator_params,
        )

    @staticmethod
    def predict_N(p: int, q: int, z: sp.Expr):
        """
        Predicts the dimension of the NxN matrices in the CMF
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
        with a_values, b_values and z_eval substituted in.
        """
        p = len(a_values)
        q = len(b_values)
        a_symbols = sp.symbols(f"a:{p}")
        b_symbols = sp.symbols(f"b:{q}")
        values = [sp.hyper(a_symbols, b_symbols, z)]
        for _ in range(1, pFq.predict_N(p, q, z_eval)):
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
        a_values: list[sp.Rational], b_values: list[sp.Rational], z: sp.Rational
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
        vector = pFq.state_vector(a_anchor, b_anchor, z)
        m = pFq.contiguous_relation((a_values, b_values), (a_anchor, b_anchor), z)
        return (vector * m)[0]
