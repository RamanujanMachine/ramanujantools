import sympy as sp
from sympy.abc import n, z

from ramanujantools.cmf import CMF
from ramanujantools.cmf import pFq
from ramanujantools import Matrix, Position
from ramanujantools.cmf.d_finite import theta

"""
there are eight cases to check for the hardcoded formula in pFq.determinant (the p=q+1 z=1 case is trickiest).
the hardcoded implementation pFq.determinant gives the best performance.
pFq_determinant_from_char_poly implementation is less prone to human errors but still quicker than a direct calculation.
We test pFq_determinant_from_char_poly against the actual det() for cmf matrices and then test the hardcoded version against the char_poly calculation.
"""
def pFq_determinant_from_char_poly(p, q, z, axis: sp.Symbol):
    is_y_shift = True if axis.name.startswith("y") else False
    shift_coeff = axis - 1 if axis.name.startswith("y") else axis
    S = sp.symbols("S")   #represents a shift operator

    #generate the characteristic polynomial for S
    theta_subs = shift_coeff * S - shift_coeff
    differential_equation = pFq.differential_equation(p, q, z).subs({z: z})
    char_poly_for_S_operator = sp.monic( differential_equation.subs({theta: theta_subs}),
                               S)
    free_coeff = char_poly_for_S_operator.coeff_monomial(1)

    matrix_dim = char_poly_for_S_operator.degree()

    if is_y_shift:
        return sp.factor((((-1) ** matrix_dim) / free_coeff).subs({axis: axis + 1}))
    else:
        return sp.factor((-1) ** matrix_dim * free_coeff)

def test_determinant_from_char_poly_32_z1_xaxis():
    p = 3
    q = 2
    z = 1
    axis = sp.symbols("x0")
    cmf = pFq(p, q,z)
    mat = cmf.matrices[axis] #we expect char poly to return factored form. should name reflect this?
    cmf_det = sp.factor(mat.det())
    calculated_from_char_poly = pFq_determinant_from_char_poly(p,q,z,axis)
    assert cmf_det == calculated_from_char_poly

def test_determinant_from_char_poly_32_z_xaxis():
    p = 3
    q = 2
    z = sp.symbols("z")
    axis = sp.symbols("x0")
    cmf = pFq(p, q,z)
    mat = cmf.matrices[axis] #we expect char poly to return factored form. should name reflect this?
    cmf_det = sp.factor(mat.det())
    calculated_from_char_poly = pFq_determinant_from_char_poly(p,q,z,axis)
    assert cmf_det == calculated_from_char_poly

def test_determinant_from_char_poly_43_z1_xaxis():
        p = 4
        q = 3
        z = 1
        axis = sp.symbols("x0")
        cmf = pFq(p, q,z)
        mat = cmf.matrices[axis] #we expect char poly to return factored form. should name reflect this?
        cmf_det = sp.factor(mat.det())
        calculated_from_char_poly = pFq_determinant_from_char_poly(p,q,z,axis)
        assert cmf_det == calculated_from_char_poly

def test_determinant_from_char_poly_43_z_xaxis():
    p = 4
    q = 3
    z = sp.symbols("z")
    axis = sp.symbols("x0")
    cmf = pFq(p, q, z)
    mat = cmf.matrices[axis]  # we expect char poly to return factored form. should name reflect this?
    cmf_det = sp.factor(mat.det())
    calculated_from_char_poly = pFq_determinant_from_char_poly(p, q, z, axis)
    assert cmf_det == calculated_from_char_poly

def test_determinant_from_char_poly_32_z1_yaxis():
    p = 3
    q = 2
    z = 1
    axis = sp.symbols("y0")
    cmf = pFq(p, q,z)
    mat = cmf.matrices[axis] #we expect char poly to return factored form. should name reflect this?
    cmf_det = sp.factor(mat.det())
    calculated_from_char_poly = pFq_determinant_from_char_poly(p,q,z,axis)
    assert cmf_det == calculated_from_char_poly

def test_determinant_from_char_poly_32_z_yaxis():
    p = 3
    q = 2
    z = sp.symbols("z")
    axis = sp.symbols("y0")
    cmf = pFq(p, q,z)
    mat = cmf.matrices[axis] #we expect char poly to return factored form. should name reflect this?
    cmf_det = sp.factor(mat.det())
    calculated_from_char_poly = pFq_determinant_from_char_poly(p,q,z,axis)
    assert cmf_det == calculated_from_char_poly

def test_determinant_from_char_poly_24_z_xaxis():
    p = 2
    q = 4
    z = sp.symbols("z")
    axis = sp.symbols("x0")
    cmf = pFq(p, q,z)
    mat = cmf.matrices[axis] #we expect char poly to return factored form. should name reflect this?
    cmf_det = sp.factor(mat.det())
    calculated_from_char_poly = pFq_determinant_from_char_poly(p,q,z,axis)
    assert cmf_det == calculated_from_char_poly



# the formula depends on the axis (x\y), the relation between p and q, and whether z=1.
def det_test(p,q,z,axis):
    det = pFq_determinant_from_char_poly(p,q,z,axis)
    calc = pFq.determinant(p,q,z,axis)
    assert calc==det

#p=q+1
def test_det_z1_32_x0():
    det_test(3,2,1,sp.symbols("x0"))

def test_det_z_32_x0():
    det_test(3,2,sp.symbols("z"),sp.symbols("x0"))

def test_det_z1_43_x0():
    det_test(4,3,1,sp.symbols("x0"))

def test_det_z_43_x0():
    det_test(4,3,sp.symbols("z"),sp.symbols("x0"))


def test_det_z_65_x1():
    det_test(6,5,sp.symbols("z"),sp.symbols("x1"))

def test_det_z1_32_y0():
    det_test(3,2,1,sp.symbols("y0"))


def test_det_z_32_y0():
    det_test(3,2,sp.symbols("z"),sp.symbols("y0"))

def test_det_z1_65_y0():
    det_test(6,5,1,sp.symbols("y0"))

def test_det_z1_43_y1():
    det_test(4,3,1,sp.symbols("y1"))

def test_det_z_43_y0():
    det_test(4,3,sp.symbols("z"),sp.symbols("y0"))


def test_det_z1_52_x0():
    det_test(5,2,1,sp.symbols("x0"))

def test_det_z_52_x0():
    det_test(5,2,sp.symbols("z"),sp.symbols("x0"))

def test_det_z2_24_x0():
    det_test(2,4,2,sp.symbols("x0"))

def test_det_z_24_x0():
    det_test(2,4,sp.symbols("z"),sp.symbols("x0"))

def test_det_z1_53_y0():
    det_test(5,3,1,sp.symbols("y0"))


def test_det_z_26_y0():
    det_test(2,6,sp.symbols("z"),sp.symbols("y0"))

def test_det_z1_26_y0():
    det_test(2,6,1,sp.symbols("y0"))

def test_det_z_51_y0():
    det_test(5,1,sp.symbols("z"),sp.symbols("y0"))