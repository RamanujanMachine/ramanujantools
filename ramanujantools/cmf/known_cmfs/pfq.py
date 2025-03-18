import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix
from ramanujantools.cmf import CMF


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
                y[i]: Matrix(-M / (y[i] + 1) + sp.eye(equation_size)) for i in range(q)
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

        matrices = {x[i]: Matrix(M / x[i] + sp.eye(equation_size)) for i in range(p)}
        matrices.update(y_matrices)
        super().__init__(matrices=matrices, validate=False)
