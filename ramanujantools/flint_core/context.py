from functools import cache, partial

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
    return ctx_type.get([str(symbol) for symbol in sorted(symbols, key=str)], "lex")


def flint_from_sympy(poly: sp.Expr, ctx: FlintContext) -> FlintPoly:
    """
    Converts a sympy poly to a flint mpoly.
    """
    return flint_converter(ctx)(poly)


def _flint_composition(ctx: FlintContext, substitutions: dict) -> list[FlintPoly]:
    """Build FLINT compose arguments for a symbolic substitution."""
    substitutions = {str(key): value for key, value in substitutions.items()}
    return [
        generator
        if str(generator) not in substitutions
        else flint_from_sympy(substitutions[str(generator)], ctx)
        for generator in ctx.gens()
    ]


def flint_converter(ctx: FlintContext):
    """Build a cached, non-expanding SymPy-expression converter."""
    generators = {sp.Symbol(str(generator)): generator for generator in ctx.gens()}

    @cache
    def convert(expression: sp.Expr) -> FlintPoly:
        if not isinstance(expression, sp.Basic):
            expression = sp.sympify(expression)

        if expression.is_Integer:
            result = ctx.constant(int(expression))
        elif expression.is_Rational:
            result = ctx.constant(flint.fmpq(int(expression.p), int(expression.q)))
        elif expression.is_Symbol:
            result = generators[expression]
        elif expression.is_Add:
            result = sum(
                (convert(argument) for argument in expression.args), ctx.constant(0)
            )
        elif expression.is_Mul:
            result = ctx.constant(1)
            for argument in expression.args:
                result *= convert(argument)
        elif (
            expression.is_Pow
            and expression.exp.is_Integer
            and expression.exp.is_nonnegative
        ):
            result = convert(expression.base) ** int(expression.exp)
        else:
            raise sp.PolynomialError(
                "Unsupported polynomial expression node "
                f"{type(expression).__name__}: {expression.func}"
            )
        return result

    return convert


def _fmpz_poly_to_sympy(poly: flint.fmpz_poly, symbol: sp.Symbol) -> sp.Expr:
    terms = []
    for exponent, coefficient in enumerate(poly):
        coefficient = sp.Integer(int(coefficient))
        if coefficient == 0:
            continue
        if exponent == 0:
            terms.append(coefficient)
            continue
        power = symbol if exponent == 1 else sp.Pow(symbol, exponent, evaluate=False)
        terms.append(
            power if coefficient == 1 else sp.Mul(coefficient, power, evaluate=False)
        )
    return sp.Add(*terms, evaluate=False)


def _fmpz_mpoly_to_fmpz_poly(poly: flint.fmpz_mpoly) -> flint.fmpz_poly:
    coefficients = [flint.fmpz(0)] * (poly.total_degree() + 1)
    for monomial, coefficient in poly.terms():
        coefficients[monomial[0]] = coefficient
    return flint.fmpz_poly(coefficients)


def _flint_polynomial_to_sympy(poly) -> sp.Expr:
    gens = poly.context().gens()
    symbols = [sp.Symbol(str(gen)) for gen in gens]
    if len(symbols) == 1 and isinstance(poly, flint.fmpz_mpoly):
        return _fmpz_poly_to_sympy(_fmpz_mpoly_to_fmpz_poly(poly), symbols[0])

    coefficients = {
        monomial: (
            sp.Integer(int(coefficient))
            if isinstance(coefficient, flint.fmpz)
            else sp.Rational(int(coefficient.numerator), int(coefficient.denominator))
        )
        for monomial, coefficient in poly.terms()
    }
    if not coefficients:
        return sp.S.Zero
    return sp.Poly.from_dict(coefficients, *symbols).as_expr()


def flint_to_sympy(
    poly,
    factor: bool = True,
    symbol: sp.Symbol | None = None,
) -> sp.Expr:
    """
    Converts a FLINT polynomial to a SymPy expression.
    """
    if isinstance(poly, flint.fmpz_poly):
        if symbol is None:
            raise ValueError("A symbol is required for an fmpz_poly")
        convert = partial(_fmpz_poly_to_sympy, symbol=symbol)
    else:
        convert = _flint_polynomial_to_sympy
    if not factor:
        return convert(poly)

    content, factors = poly.factor()
    result = sp.sympify(content)
    for polynomial, multiplicity in factors:
        result *= convert(polynomial) ** multiplicity
    return result
