from typing import List, TypeAlias

import flint
import sympy as sp

FlintContext: TypeAlias = flint.fmpz_mpoly_ctx | flint.fmpq_mpoly_ctx
FlintPoly: TypeAlias = flint.fmpz_mpoly | flint.fmpq_mpoly_ctx


def mpoly_ctx(symbols: List[sp.Symbol], fmpz: bool) -> FlintContext:
    ctx_type = flint.fmpz_mpoly_ctx if fmpz else flint.fmpq_mpoly_ctx
    return ctx_type.get(
        [str(symbol) for symbol in list(sorted(symbols, key=str))], "lex"
    )
