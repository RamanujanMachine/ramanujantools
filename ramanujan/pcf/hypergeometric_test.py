from pytest import raises
import sympy as sp
from sympy.abc import c, n, z

from ramanujan.pcf import PCF, hyp_1f1_limit, hyp_2f1_limit


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
    pcf = PCF(n + 1, n + 1)
    limit = sp.E - 1
    assert sp.simplify(limit - hyp_1f1_limit(pcf)) == 0


def test_1f1_limit_parametric():
    pcf = PCF(n + 2 - z, z * (n + 1))
    limit = (z * (sp.exp(z) - 1)) / ((z - 1) * sp.exp(z) + 1)
    assert sp.simplify(limit - hyp_1f1_limit(pcf)) == 0


def test_2f1_limit():
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    limit = (sp.root(4, 3) + 1) / (sp.root(4, 3) - 1)
    assert sp.simplify(limit - hyp_2f1_limit(pcf).rewrite(sp.log).rewrite(sp.exp)) == 0
