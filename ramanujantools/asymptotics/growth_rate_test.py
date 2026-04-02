import pytest

import sympy as sp
from sympy.abc import n

from ramanujantools.asymptotics.growth_rate import GrowthRate


def test_equality_type_gate():
    growth_rate = GrowthRate()
    assert growth_rate is not None

    assert growth_rate.__eq__(1) == NotImplemented
    assert growth_rate.__eq__("Some String") == NotImplemented


def test_tropical_dot_product_simulation():
    """Verify the combined workflow used in the final U_inv * CFM matrix multiplication."""
    g_base = GrowthRate(polynomial_degree=2, exp_base=5)
    g_shift = GrowthRate(polynomial_degree=7, exp_base=1)

    # The Tropical Dot Product
    result = g_base * g_shift

    assert result.polynomial_degree == 9
    assert result.exp_base == 5


def test_simplification():
    """Equality must survive mathematically identical but unsimplified expressions."""
    growth_rate = GrowthRate(
        polynomial_degree=n**2 - n**2,
        sub_exp=sp.expand((n + 1) ** 2 - n**2 - 2 * n - 1),
    )

    assert GrowthRate(polynomial_degree=0) == growth_rate.simplify()


def test_add_type_gate():
    g = GrowthRate()
    assert g.__add__(3) == NotImplemented


def test_add_max_filter():
    """Addition must act as a strict max() gatekeeper using __gt__."""
    dominant = GrowthRate(factorial_power=2)
    weak = GrowthRate(factorial_power=1)

    assert dominant + weak == dominant
    assert weak + dominant == dominant


def test_add_zero_passthrough():
    """Addition with 0 or None must pass the object through untouched."""
    growth_rate = GrowthRate(factorial_power=2, exp_base=3)
    assert growth_rate + GrowthRate() == growth_rate
    assert GrowthRate() + growth_rate == growth_rate


def test_mul_type_gate():
    g = GrowthRate()
    assert g.__mul__(n**2) == NotImplemented


def test_mul_growth_combination():
    """Multiplication of two GrowthRates must cleanly combine their formal exponents."""
    g1 = GrowthRate(
        factorial_power=1,
        exp_base=2,
        sub_exp=sp.sqrt(n),
        polynomial_degree=2,
        log_power=1,
    )
    g2 = GrowthRate(
        factorial_power=2, exp_base=3, sub_exp=n, polynomial_degree=3, log_power=2
    )

    result = g1 * g2

    assert result.factorial_power == 3
    assert result.exp_base == 6
    assert sp.simplify(result.sub_exp - (sp.sqrt(n) + n)) == 0
    assert result.polynomial_degree == 5
    assert result.log_power == 3


def test_gt_level_1_factorial():
    """factorial_power (Factorial) strictly dominates all other bounds."""
    g1 = GrowthRate(factorial_power=2, exp_base=1, polynomial_degree=-100)
    g2 = GrowthRate(factorial_power=1, exp_base=1000, polynomial_degree=100)

    assert g1 > g2
    assert not (g2 > g1)


def test_gt_level_2_base_exponential():
    """exp_base (Base Exp) strictly dominates sub_exp (Fractional Exp) and polynomials."""
    # g2 has a larger base lambda, so it dominates g1 despite g1's massive fractional sub_exp
    g1 = GrowthRate(exp_base=1, sub_exp=1000 * sp.sqrt(n))
    g2 = GrowthRate(exp_base=2, sub_exp=0)

    assert g2 > g1

    # Complex magnitude check: |2i| = 2 > |1|
    g3 = GrowthRate(exp_base=sp.I * 2)
    g4 = GrowthRate(exp_base=1)
    assert g3 > g4


def test_gt_level_3_fractional_exponential():
    """sub_exp (Fractional Exp) dominates polynomials."""
    # Since lambda is tied at 1, sub_exp triggers
    g1 = GrowthRate(exp_base=1, sub_exp=sp.sqrt(n), polynomial_degree=0)
    g2 = GrowthRate(exp_base=1, sub_exp=0, polynomial_degree=1000)

    assert g1 > g2


def test_gt_level_4_polynomial():
    """polynomial_degree (Polynomial) dominates logarithmic Jordan depth."""
    # Since lambda and sub_exp are tied, polynomial_degree triggers
    g1 = GrowthRate(polynomial_degree=sp.Rational(3, 2), log_power=0)
    g2 = GrowthRate(polynomial_degree=1, log_power=10)

    assert g1 > g2


def test_gt_level_5_logarithmic():
    """Jordan depth acts as the final logarithmic tie-breaker."""
    g1 = GrowthRate(log_power=2)
    g2 = GrowthRate(log_power=1)

    assert g1 > g2


def test_gt_complex_oscillation_fallthrough():
    """
    Pure imaginary terms in sub_exp (oscillation) have a real limit of 0.
    The > operator must recognize the tie and fall through to the next level.
    """
    g1 = GrowthRate(sub_exp=sp.I * n, polynomial_degree=2)
    g2 = GrowthRate(sub_exp=0, polynomial_degree=1)

    assert g1 > g2
