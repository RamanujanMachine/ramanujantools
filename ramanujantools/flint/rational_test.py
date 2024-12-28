import flint
import sympy as sp
from sympy.abc import x, y

from ramanujantools.flint import FlintRational


def flintify(expr: sp.Expr) -> FlintRational:
    return FlintRational.from_sympy(expr)


def test_from_sympy():
    expression = (x + y - 3) / (x**2 - y)
    ctx = flint.fmpz_mpoly_ctx.get(["x", "y"], "lex")
    _x, _y = ctx.gens()
    expected = FlintRational(_x + _y - 3, _x**2 - _y)
    assert expected == flintify(expression)


def test_add():
    expr1 = x / y
    expr2 = (3 + x) / (7 - y) + y
    assert flintify(expr1 + expr2) == flintify(expr1) + flintify(expr2)


def test_sub():
    expr1 = x / y
    expr2 = (3 + x) / (7 - y) + y
    assert flintify(expr1 - expr2) == flintify(expr1) - flintify(expr2)


def test_mul():
    expr1 = x / y
    expr2 = (3 + x) / (7 - y) + y
    assert flintify(expr1 * expr2) == flintify(expr1) * flintify(expr2)


def test_mul_scalar():
    expr = x / y
    assert flintify(expr * 7) == flintify(expr) * 7


def test_rmul_scalar():
    expr = x / y
    assert flintify(7 * expr) == 7 * flintify(expr)


def test_div_scalar():
    expr = x / y
    assert flintify(expr / 7) == flintify(expr) / 7


def test_rdiv_scalar():
    expr = x / y
    assert flintify(7 / expr) == 7 / flintify(expr)


def test_eq():
    expr = x / y
    assert expr == expr / 2 + 3 - 3 + expr / 2


def test_factor():
    expected = (x + y) * (x**2 + y**2) * (y - 3) / ((x + 17) * (y - 15) * (x * y + 1))
    assert expected == flintify(expected.expand()).factor()
