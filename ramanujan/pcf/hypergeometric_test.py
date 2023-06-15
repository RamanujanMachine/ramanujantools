from pytest import raises
import sympy as sp
from sympy.abc import n, z

from ramanujan.pcf import PCF, hyp_1f1_limit


def test_1f1_bad_degree_throws():
    with raises(ValueError):
        hyp_1f1_limit(PCF(1, n))
    with raises(ValueError):
        hyp_1f1_limit(PCF(n, 1))
    with raises(ValueError):
        hyp_1f1_limit(PCF(n**2, n))
    with raises(ValueError):
        hyp_1f1_limit(PCF(n, n**2))


def test_1f1_limit():
    assert sp.E - 1 == hyp_1f1_limit(PCF(n + 1, n + 1))


def test_1f1_limit_parametric():
    limit = (z * (sp.exp(z) - 1)) / ((z - 1) * sp.exp(z) + 1)
    assert limit == hyp_1f1_limit(PCF(n + 2 - z, z * (n + 1)))
