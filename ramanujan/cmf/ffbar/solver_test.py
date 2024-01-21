import pytest
from sympy.abc import x, y

from ramanujan import GenericPolynomial
from ramanujan.cmf.ffbar import solve_ffbar, linear_condition, quadratic_condition


@pytest.mark.parametrize("deg", [1, 2])
def test_solver_full_poly(deg: int):
    f, _ = GenericPolynomial.of_combined_degree(deg=deg, var_name="c", variables=[x, y])
    fbar, _ = GenericPolynomial.of_combined_degree(
        deg=deg, var_name="d", variables=[x, y]
    )
    solutions = solve_ffbar(f, fbar)
    for f, fbar in solutions:
        assert linear_condition(f, fbar) == 0
        assert quadratic_condition(f, fbar) == 0
