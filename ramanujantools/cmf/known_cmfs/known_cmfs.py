import sympy as sp
from sympy.abc import a, b, c, x, y, z

from ramanujantools import Matrix
from ramanujantools.cmf import CMF
from ramanujantools.cmf.ffbar import FFbar

x0, x1, x2 = sp.symbols("x:3")
y0, y1 = sp.symbols("y:2")
c0, c1, c2, c3 = sp.symbols("c:4")
theta = sp.symbols("theta")


def e() -> CMF:
    return CMF(
        matrices={
            x: Matrix([[1, -y - 1], [-1, x + y + 2]]),
            y: Matrix([[0, -y - 1], [-1, x + y + 1]]),
        },
        validate=False,
    )


def pi() -> CMF:
    return CMF(
        matrices={
            x: Matrix([[x, -x], [-y, 2 * x + y + 1]]),
            y: Matrix([[1 + y, -x], [-1 - y, x + 2 * y + 1]]),
        },
        validate=False,
    )


def symmetric_pi():
    return CMF(
        matrices={
            x: Matrix([[x, x * y], [1, 1 + 2 * x + y]]),
            y: Matrix([[y, x * y], [1, 1 + x + 2 * y]]),
        },
        validate=False,
    )


def zeta3() -> CMF:
    return CMF(
        matrices={
            x: Matrix(
                [
                    [0, -(x**3)],
                    [(x + 1) ** 3, x**3 + (x + 1) ** 3 + 2 * y * (y - 1) * (2 * x + 1)],
                ]
            ),
            y: Matrix(
                [
                    [-(x**3) + 2 * x**2 * y - 2 * x * y**2 + y**3, -(x**3)],
                    [x**3, x**3 + 2 * x**2 * y + 2 * x * y**2 + y**3],
                ]
            ),
        },
        validate=False,
    )


def var_root_cmf() -> CMF:
    """
    This is not a standard f,bar(f) matrix field.
    Note that b(x,y) depends on y, while a(x) does not, and x=c/2-y is the root which depends on y
    """
    Y = 2 * y - c2
    b = -x * (x + c0) * (x + c1) * (2 * x + Y)
    a = (2 * x + c1 + 1) * (2 * x + c0 + 1) - x * (x + 1)
    F = x**2 + x * (Y + 1) + (Y + 1 - c1) * (Y + 1 - c0)
    G = -(Y + 2 * x) * (x + c1 + c0 - (Y + 1))
    return CMF(
        matrices={x: Matrix([[0, b], [1, a]]), y: Matrix([[G, b], [1, F]])},
        validate=False,
    )


def cmf1() -> CMF:
    return FFbar(f=c0 + c1 * (x + y), fbar=c2 + c3 * (x - y))


def cmf2() -> CMF:
    return FFbar(
        f=(
            (2 * c1 + c2) * (c1 + c2)
            - c3 * c0
            - c3 * ((2 * c1 + c2) * (x + y) + (c1 + c2) * (2 * x + y))
            + c3**2 * (2 * x**2 + 2 * x * y + y**2)
        ),
        fbar=c3 * (c0 + c2 * x + c1 * y) - c3**2 * (2 * x**2 - 2 * x * y + y**2),
    )


def cmf3_1() -> CMF:
    return FFbar(
        f=-((c0 + c1 * (x + y)) * (c0 * (x + 2 * y) + c1 * (x**2 + x * y + y**2))),
        fbar=(c0 + c1 * (-x + y)) * (c0 * (x - 2 * y) - c1 * (x**2 - x * y + y**2)),
    )


def cmf3_2() -> CMF:
    return FFbar(
        f=-(x + y) * (c0**2 + 2 * c1**2 * (x**2 + x * y + y**2)),
        fbar=(x - y) * (c0**2 + 2 * c1**2 * (x**2 - x * y + y**2)),
    )


def cmf3_3() -> CMF:
    return FFbar(
        f=(x + y) * (c0**2 - c0 * c1 * (x - y) - 2 * c1**2 * (x**2 + x * y + y**2)),
        fbar=(c0 + c1 * (x - y)) * (3 * c0 * (x - y) + 2 * c1 * (x**2 - x * y + y**2)),
    )


def hypergeometric_derived_2F1() -> CMF:
    return CMF(
        matrices={
            a: Matrix(
                [[1 + 2 * a, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 4 * a + 2 * b + 4 * c]]
            ),
            b: Matrix(
                [[1 + 2 * b, (1 + 2 * a) * (1 + 2 * b)], [1, 5 + 2 * a + 4 * b + 4 * c]]
            ),
            c: Matrix(
                [
                    [-1 - 2 * c, (1 + 2 * a) * (1 + 2 * b)],
                    [1, 3 + 2 * a + 2 * b + 2 * c],
                ]
            ),
        },
        validate=False,
    )


def hypergeometric_derived_3F2() -> CMF:
    Sx = x0 + x1 + x2
    Sy = y0 + y1
    Tx = x0 * x1 + x0 * x2 + x1 * x2
    Ty = y0 * y1
    Px = x0 * x1 * x2
    M = Matrix(
        [
            [0, 0, Px / ((1 - z) * z)],
            [z, 1, ((Tx + Sx + 1) * z - Ty) / ((1 - z) * z)],
            [0, z, ((Sx + 1) * z + Sy + 1) / (1 - z)],
        ],
    )
    I3 = sp.eye(3)
    return CMF(
        matrices={
            x0: M / x0 + I3,
            x1: M / x1 + I3,
            x2: M / x2 + I3,
            y0: -M / (y0 + 1) + I3,
            y1: -M / (y1 + 1) + I3,
        },
        validate=False,
    )


def pFq(
    p: int,
    q: int,
    z_eval: int = z,
    theta_derivative: bool = True,
    negate_denominator_params: bool = False,
) -> CMF:
    r"""
    Constructs the pFq CMF, derived from the differentiation property of generalized hypergeometric functions:
    https://en.wikipedia.org/wiki/Generalized_hypergeometric_function

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

    Returns:
        A pFq CMF that
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
        M = (basis_transition_matrix * M * (basis_transition_matrix.inverse())).factor()

    if negate_denominator_params:
        M = M.subs({y[i]: -y[i] for i in range(q)})
        y_matrices = {
            y[i]: Matrix(-M / (y[i] + 1) + sp.eye(equation_size)) for i in range(q)
        }
    else:
        y_matrices = {
            y[i]: Matrix(M.subs({y[i]: y[i] + 1}) / y[i] + Matrix.eye(equation_size))
            .inverse()
            .factor()
            for i in range(q)
        }

    matrices = {x[i]: Matrix(M / x[i] + sp.eye(equation_size)) for i in range(p)}
    matrices.update(y_matrices)
    return CMF(matrices=matrices, validate=False)
