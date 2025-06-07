import flint
import sympy as sp

FlintContext = flint.fmpz_mpoly_ctx | flint.fmpq_mpoly_ctx
FlintPoly = flint.fmpz_mpoly | flint.fmpq_mpoly


def flint_ctx(symbols: list[sp.Symbol], fmpz: bool) -> FlintContext:
    """
    Creates a FlintContext
    Args:
        symbols: The symbols to be supported by the FlintContext
        fmpz: if True, returns fmpz_mpoly_ctx. Otherwise returns fmpq_mpoly_ctx.
    """
    ctx_type = flint.fmpz_mpoly_ctx if fmpz else flint.fmpq_mpoly_ctx
    return ctx_type.get(
        [str(symbol) for symbol in list(sorted(symbols, key=str))], "lex"
    )


def flint_from_sympy(poly: sp.Expr, ctx: FlintContext) -> FlintPoly:
    """
    Converts a sympy expression to a flint mpoly.
    """
    gens = tuple(sp.Symbol(str(gen)) for gen in ctx.gens())
    sp_poly = sp.Poly(poly, gens)
    monoms = sp_poly.monoms()
    coeffs = sp_poly.coeffs()
    mpoly_type = type(ctx.constant(0))
    # Detect if context is integer or rational
    if "fmpz" in mpoly_type.__name__:
        monom_dict = {monom: int(coeff) for monom, coeff in zip(monoms, coeffs)}
    else:
        monom_dict = {
            monom: flint.fmpq(coeff.numerator, coeff.denominator)
            for monom, coeff in zip(monoms, coeffs)
        }
    return mpoly_type(monom_dict, ctx)


def flint_to_sympy(poly) -> sp.Expr:
    """
    Factors an mpoly polynomial and returns it as a sp.Expr
    """
    gens = poly.context().gens()
    symbols = [sp.Symbol(str(gen)) for gen in gens]
    content, factors = poly.factor()
    p = sp.simplify(content)
    for factor, multiplicity in factors:
        coeffs = factor.coeffs()
        monoms = factor.monoms()
        expr = sum(
            coeff * sp.Mul(*[sym**exp for sym, exp in zip(symbols, monom) if exp])
            for coeff, monom in zip(coeffs, monoms)
        )
        p *= expr**multiplicity
    return p
