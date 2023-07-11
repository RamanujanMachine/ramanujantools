from pytest import approx
from sympy.abc import c, n

from ramanujan.pcf import PCF, PCFFromMatrix


def test_repr():
    pcf = PCF(1 + n, 3 - n)
    assert pcf == eval(repr(pcf))


def test_degree():
    pcf = PCF(1 + n - n**2, 3 - n**9)
    assert (2, 9) == pcf.degree()


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


def test_pcf_from_matrix():
    from ramanujan.pcf import PCF
    from ramanujan.cmf.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1.subs([[c0, 0], [c1, 1], [c2, 1], [c3, 3]])
    matrix = cmf.trajectory_matrix([1, 1], [1, 1])
    assert PCFFromMatrix.convert(matrix) == PCF(5 + 10 * n, 1 - 9 * n**2)


def test_pcf_from_matrix_parametric():
    from ramanujan.pcf import PCF
    from ramanujan.cmf.known_cmfs import cmf1, c0, c1, c2, c3

    cmf = cmf1.subs([[c0, 0], [c1, 1], [c2, 1], [c3, c]])
    matrix = cmf.trajectory_matrix([1, 1], [1, 1])
    assert PCFFromMatrix.convert(matrix) == PCF((1 + 2 * n) * (c + 2), 1 - (c * n) ** 2)
