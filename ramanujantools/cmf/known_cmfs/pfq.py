from typing import List

import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix
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
        negate_denominator_params: bool = False,
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
        negate_denominator_params: bool = False,
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
            M = M.subs({y[i]: -y[i] for i in range(q)})
            y_matrices = {
                y[i]: Matrix(-M / (y[i] + 1) + Matrix.eye(equation_size))
                for i in range(q)
            }
        else:
            y_matrices = {
                y[i]: Matrix(
                    M.subs({y[i]: y[i] + 1}) / y[i] + Matrix.eye(equation_size)
                )
                .inverse()
                .factor()
                for i in range(q)
            }

        matrices = {
            x[i]: Matrix(M / x[i] + Matrix.eye(equation_size)) for i in range(p)
        }
        matrices.update(y_matrices)
        return matrices

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
    def state_vector(p: int, q: int, z_eval: sp.Expr = z):
        a_symbols = sp.symbols(f"a:{p}")
        b_symbols = sp.symbols(f"b:{q}")
        values = [sp.hyper(a_symbols, b_symbols, z).simplify()]
        for _ in range(1, pFq.predict_N(p, q, z_eval)):
            values.append(pFq.theta_derivative(values[-1]))
        return Matrix(values).transpose().subs({z: z_eval}).simplify()

    @staticmethod
    def theta_derivative(expr: sp.Expr):
        return z * sp.Derivative(expr, z).simplify()

    @staticmethod
    def list_to_dict(values: List[sp.Rational], symbol: str):
        dim = len(values)
        symbols = sp.symbols(f"{symbol}:{dim}")
        return {symbols[i]: values[i] for i in range(dim)}

    @staticmethod
    def evaluate(
        a_values: List[sp.Rational], b_values: list[sp.Rational], z: sp.Rational
    ) -> sp.Expr:
        a_values = [sp.S(value) for value in a_values]
        b_values = [sp.S(value) for value in b_values]
        p = len(a_values)
        q = len(b_values)
        a_anchor = [
            sp.sign(value) * (value - (value.floor() - 1)) for value in a_values
        ]
        b_anchor = [
            sp.sign(value) * (value - (value.floor() - 2)) for value in b_values
        ]
        anchor = pFq.list_to_dict(a_anchor, "a") | pFq.list_to_dict(b_anchor, "b")
        start = pFq.list_to_dict(a_anchor, "x") | pFq.list_to_dict(b_anchor, "y")
        end = pFq.list_to_dict(a_values, "x") | pFq.list_to_dict(b_values, "y")
        vector = pFq.state_vector(p, q, z).subs(anchor).simplify()
        m = pFq(p, q, z).work(start, end)
        return (vector * m)[0]
