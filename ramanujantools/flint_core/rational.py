from __future__ import annotations

import math
import flint
import sympy as sp

from ramanujantools import Position
from ramanujantools.flint_core import (
    FlintPoly,
    FlintContext,
    flint_from_sympy,
    flint_to_sympy,
)


class FlintRational:
    """
    Represents a rational function.
    Implemented as a numerator and denominator, reduces by gcd every step
    """

    def __init__(
        self, numerator: FlintPoly, denominator: FlintPoly, ctx: FlintContext
    ) -> FlintRational:
        self.is_integer = isinstance(numerator, flint.fmpz_mpoly)
        gcd = numerator.gcd(denominator)
        if not self.is_integer:
            content = FlintRational.fmpq_gcd(numerator.coeffs() + denominator.coeffs())
            gcd *= content
        self.numerator = numerator / gcd
        self.denominator = denominator / gcd
        self.ctx = ctx

    @staticmethod
    def from_sympy(rational: sp.Expr, ctx: FlintContext) -> FlintRational:
        r"""
        Converts a rational function given as a sympy expression to a FlintRational.
        Args:
            rational: The expression to convert to flint
            ctx: The desired mpoly context (which also defines the supported variables)
        Returns:
            A FlintRational object representing the `rational` value
        """
        numerator, denominator = rational.as_numer_denom()
        return FlintRational(
            flint_from_sympy(numerator, ctx),
            flint_from_sympy(denominator, ctx),
            ctx,
        )

    @staticmethod
    def fmpq_gcd(numbers: list[flint.fmpq]) -> flint.fmpz:
        denominator = flint.fmpz(1)
        for c in numbers:
            denominator *= c.denominator
        numerators = [c.numerator * denominator / c.denominator for c in numbers]
        gcd = math.gcd(*numerators)
        return flint.fmpq(gcd, denominator)

    def inv(self) -> FlintRational:
        """
        Returns 1 / self.
        """
        return FlintRational(self.denominator, self.numerator, self.ctx)

    def __neg__(self):
        return FlintRational(-self.numerator, self.denominator, self.ctx)

    def __add__(self, other: FlintRational) -> FlintRational:
        return FlintRational(
            self.numerator * other.denominator + self.denominator * other.numerator,
            self.denominator * other.denominator,
            self.ctx,
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
            return FlintRational(numerator, denominator, self.ctx)
        else:
            return FlintRational(self.numerator * other, self.denominator, self.ctx)

    def __rmul__(self, other) -> FlintRational:
        return self * other

    def __truediv__(self, other) -> FlintRational:
        if isinstance(other, FlintRational):
            return self * other.inv()
        return FlintRational(self.numerator, self.denominator * other, self.ctx)

    def __rtruediv__(self, other) -> FlintRational:
        return other * self.inv()

    def __repr__(self) -> str:
        return f"FlintRational({self.numerator}, {self.denominator})"

    def __eq__(self, other: FlintRational) -> bool:
        return (
            self.numerator == other.numerator and self.denominator == other.denominator
        )

    def degrees(self) -> list[int]:
        return [max(poly.degrees()) for poly in [self.numerator, self.denominator]]

    def subs(self, substitutions: dict) -> FlintRational:
        """
        Substitutes symbols in self.
        """
        substitutions = Position(
            {str(key): value for key, value in substitutions.items()}
        )
        composition = []
        for gen in self.ctx.gens():
            if str(gen) in substitutions:
                value = flint_from_sympy(substitutions[str(gen)], self.ctx)
            else:
                value = gen
            composition.append(value)
        content = (
            1
            if self.is_integer
            else substitutions.denominator_lcm() ** max(self.degrees())
        )
        return FlintRational(
            (content * self.numerator).compose(*composition),
            (content * self.denominator).compose(*composition),
            self.ctx,
        )

    def factor(self) -> sp.Expr:
        """
        Factors self and returns it as a sp.Expr
        """
        return flint_to_sympy(self.numerator) / flint_to_sympy(self.denominator)
