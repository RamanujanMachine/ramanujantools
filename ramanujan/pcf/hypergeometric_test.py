from pytest import raises
import sympy as sp
from sympy.abc import c, n, z

from ramanujan.pcf import (
    PCF,
    HypergeometricLimit,
    Hypergeometric1F1Limit,
    Hypergeometric2F1Limit,
)


def bad_degrees_throw():
    with raises(ValueError):
        HypergeometricLimit(PCF(0, n))
    with raises(ValueError):
        HypergeometricLimit(PCF(n, 0))
    with raises(ValueError):
        HypergeometricLimit(PCF(n**2, n))
    with raises(ValueError):
        HypergeometricLimit(PCF(n, n**3))


def test_1f1_limit():
    pcf = PCF(n + 1, n + 1)
    limit = sp.E - 1
    assert sp.simplify(limit - Hypergeometric1F1Limit(pcf).limit()) == 0


def test_1f1_limit_parametric():
    pcf = PCF(n + 2 - z, z * (n + 1))
    limit = (z * (sp.exp(z) - 1)) / ((z - 1) * sp.exp(z) + 1)
    assert sp.simplify(limit - Hypergeometric1F1Limit(pcf).limit()) == 0


def test_2f1_limit():
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    # limit = (sp.root(4, 3) + 1) / (sp.root(4, 3) - 1) - this is the exact same thing, but sympy struggles
    limit_as_prompt = "8 * 1/2 * Hypergeometric2F1[-1/3, 1/3, 1/2, -1/8] / Hypergeometric2F1[2/3, 4/3, 3/2, -1/8]"
    assert limit_as_prompt == Hypergeometric2F1Limit(pcf).as_mathematica_prompt()


def test_2f1_limit_parameteric():
    pcf = PCF((1 + 2 * n) * (c + 2), 1 - (c * n) ** 2)
    # limit = (sp.root(c + 1, c) + 1) / (sp.root(c + 1, c) - 1)
    print(Hypergeometric2F1Limit(pcf).as_mathematica_prompt())
    assert False
