from __future__ import annotations
from typing import List, Dict

import flint
import sympy as sp


class FlintRational:
    def __init__(
        self, numerator: flint.fmpz_mpoly, denominator: flint.fmpz_mpoly
    ) -> FlintRational:
        gcd = numerator.gcd(denominator)
        self.numerator = numerator / gcd
        self.denominator = denominator / gcd

    @staticmethod
    def from_sympy(rational: sp.Expr, symbols: List = None) -> FlintRational:
        symbols = symbols or list(sorted(rational.free_symbols, key=str))
        symbols = [str(symbol) for symbol in symbols]
        ctx = flint.fmpz_mpoly_ctx.get(symbols, "lex")
        assert len(ctx.gens()) == len(symbols)
        for i in range(len(symbols)):
            exec(f"{symbols[i]} = ctx.gens()[i]")
        numerator, denominator = rational.as_numer_denom()
        return FlintRational(
            flint.fmpz_mpoly(str(numerator).replace("**", "^"), ctx),
            flint.fmpz_mpoly(str(denominator).replace("**", "^"), ctx),
        )

    def inv(self) -> FlintRational:
        return FlintRational(self.denominator, self.numerator)

    def __neg__(self):
        return FlintRational(-self.numerator, self.denominator)

    def __add__(self, other: FlintRational) -> FlintRational:
        return FlintRational(
            self.numerator * other.denominator + self.denominator * other.numerator,
            self.denominator * other.denominator,
        )

    def __radd__(self, other) -> FlintRational:
        return other + self

    def __sub__(self, other) -> FlintRational:
        return self + (-other)

    def __rsub__(self, other) -> FlintRational:
        return -self + other

    def __mul__(self, other) -> FlintRational:
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return FlintRational(numerator, denominator)

    def __rmul__(self, other) -> FlintRational:
        return FlintRational(other) * self

    def __truediv__(self, other) -> FlintRational:
        return self * other.inv()

    def __rtruediv__(self, other) -> FlintRational:
        return FlintRational(other) / self

    def __repr__(self) -> str:
        return f"FlintRational({self.numerator}, {self.denominator})"

    def subs(self, substitutions: Dict) -> FlintRational:
        substitutions = {str(key): value for key, value in substitutions.items()}
        return FlintRational(
            self.numerator.subs(substitutions), self.denominator.subs(substitutions)
        )

    @staticmethod
    def factor_poly(poly) -> sp.Expr:
        gens = poly.context().gens()
        for gen in gens:
            exec(f"{gen} = sp.Symbol('{gen}')")
        content, factors = poly.factor()
        p = sp.simplify(content)
        for factor, multiplicity in factors:
            factor = str(factor).replace("^", "**")  # from flint syntax back to python
            p *= eval(f"({factor}) ** ({multiplicity})")
        return p

    def factor(self) -> sp.Expr:
        return FlintRational.factor_poly(self.numerator) / FlintRational.factor_poly(
            self.denominator
        )
