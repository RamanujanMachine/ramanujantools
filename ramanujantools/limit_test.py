from pytest import approx

from mpmath import mp

from ramanujantools import Limit
from .limit import most_round_in_range


def test_equality():
    base_limit = Limit([[1, 2], [3, 4]])
    limit1 = base_limit * 17
    limit2 = base_limit * 31
    assert limit1 == limit2


def test_as_rational():
    p = 2
    q = 3
    limit = Limit([[0, p], [1, q]])
    assert [p, q] == limit.as_rational()


def test_as_rational_higher_order():
    p = 2
    q = 3
    limit = Limit([[1, 2, p], [3, 4, q], [5, 6, 7]])
    assert [p, q] == limit.as_rational()


def test_as_float():
    p = 2
    q = 3
    limit = Limit([[0, p], [1, q]])
    assert p / q == approx(limit.as_float(), 1e-7)


def test_precision_exact():
    a = 5
    b = a - 1
    desired_error = 5
    denominator = 10**desired_error
    limit = Limit([[a, b], [denominator, denominator]])
    assert desired_error == limit.precision()


def test_precision_floor():
    a = 5
    b = a - 2
    desired_error = 5
    denominator = 10**desired_error
    limit = Limit([[a, b], [denominator, denominator]])
    assert desired_error - 1 == limit.precision()


def test_rounding_whole_number():
    num = 0.9999999
    err = 5 * 1e-5
    assert "1" == most_round_in_range(num, err)


def test_rounding_small_change():
    num = mp.mpf(0.9999999)
    err = mp.mpf(5 * 1e-15)
    assert "0.9999998" == most_round_in_range(num, err)
