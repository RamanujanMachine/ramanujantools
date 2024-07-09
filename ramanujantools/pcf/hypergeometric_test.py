from pytest import approx, raises
import sympy as sp
from sympy.abc import n, z

from ramanujantools.pcf import (
    PCF,
    HypergeometricLimit,
    Hypergeometric1F1Limit,
    Hypergeometric2F1Limit,
)


def test_factory_bad_degrees_throw():
    with raises(ValueError):
        HypergeometricLimit(PCF(0, n))
    with raises(ValueError):
        HypergeometricLimit(PCF(n, 0))
    with raises(ValueError):
        HypergeometricLimit(PCF(n**2, n))
    with raises(ValueError):
        HypergeometricLimit(PCF(n, n**3))


def test_factory_types():
    assert isinstance(HypergeometricLimit(PCF(n, n)), Hypergeometric1F1Limit)
    assert isinstance(HypergeometricLimit(PCF(n, n**2)), Hypergeometric2F1Limit)


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
    limit = (sp.root(4, 3) + 1) / (sp.root(4, 3) - 1)
    assert float(limit - Hypergeometric2F1Limit(pcf).limit()) < 10e-50


def test_2f1_limit_parameteric():
    # going for positive c here to avoid piecewise condition of c!=-2
    c = sp.Symbol("c", positive=True)
    pcf = PCF((1 + 2 * n) * (c + 2), 1 - (c * n) ** 2)
    # limit = (sp.root(c + 1, c) + 1) / (sp.root(c + 1, c) - 1)
    hyp = Hypergeometric2F1Limit(pcf)
    assert hyp.alpha1 == -1 / c
    assert hyp.alpha2 == 1 / c
    assert hyp.beta == sp.Rational(1, 2)
    assert hyp.z == (-c + 2 * sp.sqrt(c + 1) - 2) / (4 * sp.sqrt(c + 1))


def test_1f1_limit_numerically():
    pcf = PCF(1 + 2 * n, 3 + 4 * n)
    assert pcf.limit(1000).as_float() == approx(
        float(Hypergeometric1F1Limit(pcf).limit())
    )


def test_2f1_limit_numerically():
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    assert pcf.limit(1000).as_float() == approx(
        float(Hypergeometric2F1Limit(pcf).limit())
    )


def test_1f1_subs():
    from sympy.abc import c

    pcf = PCF(c * n, 3 - n)
    hyp = Hypergeometric1F1Limit(pcf)
    assert hyp.subs(c, 7) == Hypergeometric1F1Limit(pcf.subs(c, 7))


def test_2f1_subs():
    from sympy.abc import c

    pcf = PCF((1 + 2 * n) * (c + 2), 1 - (c * n) ** 2)
    hyp = Hypergeometric2F1Limit(pcf)
    assert hyp.subs(c, 7) == Hypergeometric2F1Limit(pcf.subs(c, 7))
