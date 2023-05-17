from pytest import approx, raises
from sympy.abc import n

from cmf import Matrix, PCF, is_pcf


def test_non_pcf_throws():
    with raises(Exception):
        PCF(Matrix([[0, 0], [0, 0]]))
    with raises(Exception):
        PCF(Matrix([[1, 1], [1, 1]]))


def test_pcf_limit():
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    expected = (4 ** (1 / 3) + 1) / (4 ** (1 / 3) - 1)
    assert expected == approx(pcf.limit(100), 1e-4)
