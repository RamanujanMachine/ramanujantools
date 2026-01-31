import sympy as sp
from sympy.abc import n, z
import pytest

from ramanujantools.cmf import pFq
from ramanujantools.cmf.d_finite import theta

x0, x1 = sp.symbols("x:2")
y0, y1 = sp.symbols("y:2")


def pFq_determinant_from_char_poly(p, q, z, axis: sp.Symbol):
    """
    there are eight cases to check for the hardcoded formula in pFq.determinant (the p=q+1 z=1 case is trickiest).
    the hardcoded implementation pFq.determinant gives the best performance.
    pFq_determinant_from_char_poly implementation is less prone to human errors but still quicker than a direct calculation.
    We test pFq_determinant_from_char_poly against the actual det() for cmf matrices and then test the hardcoded version against the char_poly calculation.
    """
    is_y_shift = True if axis.name.startswith("y") else False
    shift_coeff = axis - 1 if axis.name.startswith("y") else axis
    S = sp.symbols("S")  # represents a shift operator

    # generate the characteristic polynomial for S
    theta_subs = shift_coeff * S - shift_coeff
    differential_equation = pFq.differential_equation(p, q, z)
    char_poly_for_S_operator = sp.monic(
        differential_equation.subs({theta: theta_subs}), S
    )
    free_coeff = char_poly_for_S_operator.coeff_monomial(1)
    matrix_dim = char_poly_for_S_operator.degree()

    if is_y_shift:
        return sp.factor((((-1) ** matrix_dim) / free_coeff).subs({axis: axis + 1}))
    else:
        return sp.factor((-1) ** matrix_dim * free_coeff)


@pytest.mark.parametrize(
    "p, q, z, axis",
    [
        (3, 2, 1, x0),
        (3, 2, z, x0),
        (4, 3, 1, x0),
        (4, 3, z, x0),
        (3, 2, 1, y0),
        (3, 2, z, y0),
        (2, 4, z, x0),
    ],
)
def test_determinant_against_char_poly(p, q, z, axis):
    """Tests if the manual char_poly calculation matches the actual matrix determinant."""
    cmf = pFq(p, q, z)
    mat = cmf.matrices[axis]
    cmf_det = sp.factor(mat.det())
    calculated_from_char_poly = pFq_determinant_from_char_poly(p, q, z, axis)
    assert cmf_det == calculated_from_char_poly


@pytest.mark.parametrize(
    "p, q, z, axis",
    [
        # p = q + 1 cases
        (3, 2, 1, x0),
        (3, 2, z, x0),
        (4, 3, 1, x0),
        (4, 3, z, x0),
        (6, 5, z, x1),
        (3, 2, 1, y0),
        (3, 2, z, y0),
        (6, 5, 1, y0),
        (4, 3, 1, y1),
        (4, 3, z, y0),
        # Other p, q relations
        (5, 2, 1, x0),
        (5, 2, z, x0),
        (2, 4, 2, x0),
        (2, 4, z, x0),
        (5, 3, 1, y0),
        (2, 6, z, y0),
        (2, 6, 1, y0),
        (5, 1, z, y0),
    ],
)
def test_hardcoded_determinant_formula(p, q, z, axis):
    """Tests the high-performance pFq.determinant against the char_poly method."""
    det = pFq_determinant_from_char_poly(p, q, z, axis)
    calc = pFq.determinant(p, q, z, axis)
    assert calc == det
