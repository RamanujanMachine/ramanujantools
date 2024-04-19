from pytest import approx

from ramanujan import Limit


def test_as_rational():
    p = 2
    q = 3
    limit = Limit([[0, p], [0, q]])
    assert [p, q] == limit.as_rational()


def test_as_float():
    p = 2
    q = 3
    limit = Limit([[0, p], [0, q]])
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
