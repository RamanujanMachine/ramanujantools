from ramanujantools import IntegerRelation


def test_rational_relation():
    ir = IntegerRelation([[-1, 2, -3, 4]])
    assert "IntegerRelation([[-1, 2, -3, 4]])" == repr(ir)
    assert "0 = -p0 + 2*p1 - 3*p2 + 4*p3" == str(ir)


def test_irrational_relation():
    ir = IntegerRelation([[-1, 2, -3, 4], [5, -6, 7, -8]])
    assert "IntegerRelation([[-1, 2, -3, 4], [5, -6, 7, -8]])" == repr(ir)
    assert "0 = -p0 + 2*p1 - 3*p2 + 4*p3 - L*(5*p0 - 6*p1 + 7*p2 - 8*p3)" == str(ir)


def test_negated_equality():
    ir = IntegerRelation([[1, 2], [-3, -4]])
    assert ir == ir
    assert IntegerRelation([[-1, -2], [3, 4]]) == ir
