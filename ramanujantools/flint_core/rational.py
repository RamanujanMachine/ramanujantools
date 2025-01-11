from __future__ import annotations
from typing import List, Dict

import math
import flint
import sympy as sp

from ramanujantools import Position


class FlintRational:
    """
    Represents a rational function.
    Implemented as a numerator and denominator, reduces by gcd every step
    """

    def __init__(
        self, numerator: flint.fmpq_mpoly, denominator: flint.fmpq_mpoly
    ) -> FlintRational:
        gcd = numerator.gcd(denominator) * FlintRational.content_gcd(
            numerator, denominator
        )
        self.numerator = numerator / gcd
        self.denominator = denominator / gcd

    @staticmethod
    def content_gcd(poly1, poly2):
        coeffs = poly1.coeffs() + poly2.coeffs()
        coeffs = [c.numerator for c in coeffs]
        return math.gcd(*coeffs)

    @staticmethod
    def fmpq_from_sympy(poly: sp.Expr, ctx) -> flint.fmpq_mpoly:
        r"""
        Converts a sympy expression to fmpq_mpoly.
        """
        return flint.fmpq_mpoly(str(poly).replace("**", "^"), ctx)

    @staticmethod
    def from_sympy(rational: sp.Expr, symbols: List = None) -> FlintRational:
        r"""
        Converts a rational function given as a sympy expression to a FlintRational.
        """
        symbols = symbols or list(sorted(rational.free_symbols, key=str))
        symbols = [str(symbol) for symbol in symbols]
        ctx = flint.fmpq_mpoly_ctx.get(symbols, "lex")
        assert len(ctx.gens()) == len(symbols)
        numerator, denominator = rational.as_numer_denom()
        return FlintRational(
            FlintRational.fmpq_from_sympy(numerator, ctx),
            FlintRational.fmpq_from_sympy(denominator, ctx),
        )

    def inv(self) -> FlintRational:
        """
        Returns 1 / self.
        """
        return FlintRational(self.denominator, self.numerator)

    def __neg__(self):
        return FlintRational(-self.numerator, self.denominator)

    def __add__(self, other: FlintRational) -> FlintRational:
        return FlintRational(
            self.numerator * other.denominator + self.denominator * other.numerator,
            self.denominator * other.denominator,
        )

    def __radd__(self, other: FlintRational) -> FlintRational:
        return self + other

    def __sub__(self, other) -> FlintRational:
        return self + (-other)

    def __rsub__(self, other) -> FlintRational:
        return -self + other

    def __mul__(self, other) -> FlintRational:
        if isinstance(other, FlintRational):
            numerator = self.numerator * other.numerator
            denominator = self.denominator * other.denominator
            return FlintRational(numerator, denominator)
        else:
            return FlintRational(self.numerator * other, self.denominator)

    def __rmul__(self, other) -> FlintRational:
        return self * other

    def __truediv__(self, other) -> FlintRational:
        if isinstance(other, FlintRational):
            return self * other.inv()
        return FlintRational(self.numerator, self.denominator * other)

    def __rtruediv__(self, other) -> FlintRational:
        return other * self.inv()

    def __repr__(self) -> str:
        return f"FlintRational({self.numerator}, {self.denominator})"

    def __eq__(self, other: FlintRational) -> bool:
        return (
            self.numerator == other.numerator and self.denominator == other.denominator
        )

    def degrees(self) -> List[int]:
        return [max(poly.degrees()) for poly in [self.numerator, self.denominator]]

    def subs(self, substitutions: Dict) -> FlintRational:
        """
        Substitutes symbols in self.
        """
        substitutions = Position(
            {str(key): value for key, value in substitutions.items()}
        )
        ctx = self.numerator.context()
        composition = []
        content = substitutions.denominator_lcm() ** max(self.degrees())
        for gen in ctx.gens():
            if str(gen) in substitutions:
                value = FlintRational.fmpq_from_sympy(substitutions[str(gen)], ctx)
            else:
                value = gen
            composition.append(value)
        return FlintRational(
            (content * self.numerator).compose(*composition),
            (content * self.denominator).compose(*composition),
        )

    @staticmethod
    def factor_poly(poly) -> sp.Expr:
        """
        Factors a polynomial of type fmpq_mpoly and returns it as a sp.Expr
        """
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
        """
        Factors self and returns it as a sp.Expr
        """
        return FlintRational.factor_poly(self.numerator) / FlintRational.factor_poly(
            self.denominator
        )
