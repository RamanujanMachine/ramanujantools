import pytest
import random
from sympy.abc import x, y

from ramanujantools import Matrix
from ramanujantools.cmf import CMF
from ramanujantools.cmf.known_cmfs import e, pi, zeta3
from ramanujantools.solvers import CoboundarySolver


known_cmf_list = [e(), pi(), zeta3()]


def verify_solution(mx1: Matrix, mx2: Matrix, solution: Matrix, deg: int):
    r"""
    Checks if the vector space of polynomial matrices m(x) of degree at most 'deg' satisfying
    mx1 * m(x+1) = m(x) * mx2
    contain the given solution.
    Note that m(x)=0 is always a solution.
    """
    result = CoboundarySolver.find_coboundary(mx1, mx2, deg, x)
    assert result is not None

    mm, variables = result
    # Check that the given solution appears in the vector space of solutions 'mm'
    assignment = CoboundarySolver.solve_polynomial_matrix(
        mm - solution, symbol=x, variables=variables
    )
    assert len(assignment) > 0


@pytest.mark.parametrize(
    "matrix",
    [
        Matrix([[1, x], [x**2, x + 5]]),
        Matrix([[1 + x + x**2, x], [x**5, x + 5 + x**3]]),
    ],
)
def test_self_coboundary(matrix: Matrix):
    """
    Every matrix is coboundary equivalent to itself via the identity matrix.
    """
    verify_solution(
        mx1=matrix,
        mx2=matrix,
        solution=Matrix(
            [[1, 0], [0, 1]]
        ),  # TODO : create Matrix.ID2() method which returns a 2x2 identity matrix
        deg=1,
    )


@pytest.mark.parametrize("cmf", known_cmf_list)
def test_2_lines_cmf_coboundary(cmf: CMF):
    # coboundary random two lines in a given cmf
    line = random.randint(1, 10)
    verify_solution(
        mx1=cmf.M(x)({y: line}),
        mx2=cmf.M(x)({y: line + 1}),
        solution=cmf.M(y)({y: line}),
        deg=5,
    )


@pytest.mark.parametrize("cmf", known_cmf_list)
def test_cmf_coboundary(cmf: CMF):
    r"""
    Given a family of parametrized matrices, we can find if they are part of a cmf
    """
    verify_solution(mx1=cmf.M(x), mx2=cmf.M(x)({y: y + 1}), solution=cmf.M(y), deg=6)


def test_specific_coboundary():
    mx1 = Matrix([[0, -(x**8)], [1, x**4 + (1 + x) ** 4]])
    mx2 = Matrix([[0, -(x**8)], [1, x**4 + (1 + x) ** 4 + 2 * (x**2 + (1 + x) ** 2)]])
    solution = Matrix(
        [[x**4 * (1 - 2 * x), -(x**8) * (1 + 2 * x)], [2 * x - 1, x**4 * (1 + 2 * x)]]
    )

    verify_solution(mx1=mx1, mx2=mx2, solution=solution, deg=10)


# TODO: Add negative tests, for matrices which are not polynomially coboundary equivalent
