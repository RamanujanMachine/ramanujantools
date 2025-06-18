from pytest import approx

import sympy as sp
from mpmath import mp

from ramanujantools import Matrix, Limit
from .limit import most_round_in_range


def limit_for_tests(matrix: Matrix) -> Limit:
    return Limit(matrix, matrix)


def test_as_rational():
    p = 2
    q = 3
    limit = limit_for_tests(Matrix([[0, p], [1, q]]))
    assert sp.Rational(p, q) == limit.as_rational()


def test_as_rational_higher_order():
    p = 2
    q = 3
    limit = limit_for_tests(Matrix([[1, 2, p], [3, 4, q], [5, 6, 7]]))
    assert sp.Rational(p, q) == limit.as_rational()


def test_as_float():
    p = 2
    q = 3
    limit = limit_for_tests(Matrix([[0, p], [1, q]]))
    assert p / q == approx(limit.as_float(), 1e-7)


def test_precision_exact():
    a = 5
    b = a - 1
    desired_error = 5
    denominator = 10**desired_error
    limit = Limit(
        Matrix([[a, b], [denominator, denominator]]), Matrix([[0, a], [1, denominator]])
    )
    assert desired_error == limit.precision()


def test_precision_floor():
    a = 5
    b = a - 2
    desired_error = 5
    denominator = 10**desired_error
    limit = Limit(
        Matrix([[a, b], [denominator, denominator]]), Matrix([[0, a], [1, denominator]])
    )
    assert desired_error - 1 == limit.precision()


def test_rounding_in_range():
    for num in [0.49, 0.4999999, 0.5, 0.500001, 0.51]:
        err = 5 * 1e-2
        rounded = eval(most_round_in_range(num, err))

        assert abs(num - rounded) < err


def test_rounding_whole_number():
    num = 0.9999999
    err = 5 * 1e-5
    assert "1" == most_round_in_range(num, err)


def test_rounding_small_change():
    num = mp.mpf(0.9999999)
    err = mp.mpf(5 * 1e-15)
    assert "0.9999999" == most_round_in_range(num, err)


def test_repr():
    limit = limit_for_tests(Matrix([[1, 2], [3, 4]]))
    assert limit == eval(repr(limit))
