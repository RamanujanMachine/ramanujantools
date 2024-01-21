import pytest

from ramanujan.cmf.ffbar import (
    full_poly,
    solve_ffbar,
    linear_condition,
    quadratic_condition,
)


@pytest.mark.parametrize("deg", [1, 2])
def test_solver_full_poly(deg: int):
    f = full_poly("c", deg)
    fbar = full_poly("d", deg)
    solutions = solve_ffbar(f, fbar)
    for f, fbar in solutions:
        assert linear_condition(f, fbar) == 0
        assert quadratic_condition(f, fbar) == 0
