import flint
import sympy as sp

FlintContext = flint.fmpz_mpoly_ctx | flint.fmpq_mpoly_ctx
FlintPoly = flint.fmpz_mpoly | flint.fmpq_mpoly_ctx


def mpoly_ctx(symbols: list[sp.Symbol], fmpz: bool) -> FlintContext:
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
