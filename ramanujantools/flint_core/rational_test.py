import sympy as sp
from sympy.abc import x, y

from ramanujantools.flint_core import flint_ctx, FlintRational


def flintify(expr: sp.Expr, symbols: list = None, fmpz=True) -> FlintRational:
    ctx = flint_ctx(symbols or list(expr.free_symbols), fmpz)
    return FlintRational.from_sympy(expr, ctx)


def test_from_sympy():
    expression = (x + y - 3) / (x**2 - y)
    ctx = flint_ctx(["x", "y"], fmpz=True)
    _x, _y = ctx.gens()
    expected = FlintRational(_x + _y - 3, _x**2 - _y, ctx)
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


def test_subs_integer():
    expr = (x**2 + 3 * y / 2 - sp.Rational(5, 4)) / (x * y + y**2 / 3)
    subs = {x: 3, y: x}
    assert flintify(expr.subs(subs), symbols=[x, y]) == flintify(expr).subs(subs)


def test_subs_rational():
    expr = (x**2 + 3 * y / 2 - sp.Rational(5, 4)) / (x * y + y**2 / 3)
    subs = {x: sp.Rational(2, 7), y: (x + 5) / 4}
    assert flintify(expr.subs(subs), symbols=[x, y], fmpz=False) == flintify(
        expr, fmpz=False
    ).subs(subs)
