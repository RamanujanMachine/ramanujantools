from ramanujan import Vector, Matrix


def test_vector_creation():
    assert Vector([1, 2]) == Matrix([[1], [2]])


def test_zero_vector():
    assert Vector.zero() == Vector([0, 1])


def test_inf_vector():
    assert Vector.inf() == Vector([1, 0])
