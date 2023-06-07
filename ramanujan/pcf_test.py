from pytest import approx, raises
from sympy.abc import c, n

from ramanujan import Matrix, PCF


def test_from_matrix_throws():
    with raises(ValueError):
        PCF.from_matrix(Matrix([[0, 0], [0, 0]]))
    with raises(ValueError):
        PCF.from_matrix(Matrix([[1, 1], [1, 1]]))


def test_repr():
    pcf = PCF(1 + n, 3 - n)
    assert pcf == eval(repr(pcf))


def test_limit():
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    expected = (4 ** (1 / 3) + 1) / (4 ** (1 / 3) - 1)
    assert expected == approx(pcf.limit(100), 1e-4)


def test_inflate_constant():
    a_n = n + 4
    b_n = n**2
    pcf = PCF(a_n, b_n)
    assert PCF(c * a_n, c**2 * b_n) == pcf.inflate(c)


def test_inflate_poly():
    a_n = n + 4
    b_n = n**2
    c_n = n**7 + 5 * n - 3
    pcf = PCF(a_n, b_n)
    assert PCF(c_n * a_n, c_n.subs(n, n - 1) * c_n * b_n) == pcf.inflate(c_n)


def test_deflate_constant():
    a_n = n + 4
    b_n = n**2
    pcf = PCF(c * a_n, c**2 * b_n)
    assert PCF(a_n, b_n) == pcf.deflate(c)


def test_deflate_poly():
    a_n = n + 4
    b_n = n**2
    c_n = n**7 + 5 * n - 3
    pcf = PCF(c_n * a_n, c_n.subs(n, n - 1) * c_n * b_n)
    assert PCF(a_n, b_n) == pcf.deflate(c_n)


def test_deflate_all():
    c_n = c**2 * (7 * n - 13 * c) * (2 * n - 5) ** 4 * (3 * n + 11)
    a_n = n + c
    b_n = n**2 - c * n
    pcf = PCF(c_n * a_n, c_n.subs(n, n - 1) * c_n * b_n)
    assert PCF(a_n, b_n) == pcf.deflate_all()
