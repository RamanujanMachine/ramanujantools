import pytest
import random
from sympy.abc import x, y

from ramanujan.cmf.cmf import CMF
from ramanujan.cmf import known_cmfs
from ramanujan.cmf.coboundary import CoboundarySolver
from ramanujan.matrix import Matrix
from ramanujan.simplify_object import simplify


known_cmf_list = [known_cmfs.e(), known_cmfs.pi(), known_cmfs.zeta3()]


@pytest.mark.parametrize("cmf", known_cmf_list)
def test_cmf_coboundary(cmf: CMF):
    line = random.randint(1, 10)
    mx1 = cmf.Mx({y: line})
    mx2 = cmf.Mx({y: line+1})
    result = CoboundarySolver.find_coboundary(mx1, mx2, 5, x)
    assert result is not None


def test_specific_coboundary():
    mx1 = Matrix([[0, -x**8], [1, x**4 + (1+x)**4]])
    mx2 = Matrix([[0, -x**8], [1, x**4 + (1+x)**4 + 2*(x**2 + (1+x)**2)]])
    result = CoboundarySolver.find_coboundary(mx1, mx2, 10, x)
    assert result is not None

    mm, symbols = result
    # TODO: Add projective equality between two matrices, and use it instead
    assert simplify(mx1 * mm({x:x+1}) - mm * mx2) == Matrix([[0, 0], [0, 0]])

# TODO: Add negative tests, for matrices which are not polynomially coboundary equivalent
