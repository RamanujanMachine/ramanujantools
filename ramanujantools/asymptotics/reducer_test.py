from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.asymptotics.reducer import Reducer


def test_exponential_separation():
    # We want a solution that grows like: 2^n * n^3  and  4^n * n^5
    Lambda_expected = Matrix.diag(2, 4)
    D_expected = Matrix.diag(3, 5)

    # The canonical M(n) for this is Lambda * (I + D/n)
    M_canonical = Lambda_expected * (Matrix.eye(2) + D_expected / n)

    # A rational gauge to scramble it
    U = Matrix.eye(2) + Matrix([[1, -2], [3, 1]]) / n
    M = M_canonical.coboundary(U)

    # Run the Reducer
    reducer = Reducer(M, precision=5)
    fact_power, Lambda_calc, D_calc = reducer.reduce()

    assert fact_power == 0
    assert Lambda_calc == Lambda_expected
    assert D_calc == D_expected
