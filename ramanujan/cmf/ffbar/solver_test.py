from ramanujan.cmf.ffbar import full_poly, solve_ffbar


def test_solver():
    f = full_poly("c", 2)
    fbar = full_poly("d", 2)
    solutions = solve_ffbar(f, fbar)
    for f, fbar in solutions:
        print(f, ", ", fbar)
    assert False
