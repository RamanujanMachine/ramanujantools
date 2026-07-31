from __future__ import annotations

from dataclasses import dataclass
import math
import flint
import sympy as sp

from ramanujantools import Position
from ramanujantools.flint_core.context import (
    FlintPoly,
    FlintContext,
    _flint_composition,
    flint_converter,
    flint_to_sympy,
)


def _divisible_pair(left, right):
    if not all(
        isinstance(value, flint.fmpz_poly) and value != 0 for value in (left, right)
    ):
        return None
    larger, smaller = (
        (left, right) if left.degree() >= right.degree() else (right, left)
    )
    return (larger, smaller) if divmod(larger, smaller)[1] == 0 else None


def _polynomial_gcd(left, right):
    pair = _divisible_pair(left, right)
    return pair[1] if pair else left.gcd(right)


def _polynomial_lcm(left, right):
    if left == right:
        return left
    pair = _divisible_pair(left, right)
    if pair:
        return pair[0]
    return left * (right / left.gcd(right))


@dataclass(frozen=True)
class _PolynomialFraction:
    numerator: object
    denominator: object
    scalar_shifts: tuple[int, ...] = ()

    def __mul__(self, other: _PolynomialFraction) -> _PolynomialFraction:
        left_common = _polynomial_gcd(self.numerator, other.denominator)
        right_common = _polynomial_gcd(other.numerator, self.denominator)
        return _PolynomialFraction(
            (self.numerator / left_common) * (other.numerator / right_common),
            (self.denominator / right_common) * (other.denominator / left_common),
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
        gcd = _polynomial_gcd(numerator, denominator)
        if not self.is_integer:
            content = FlintRational.fmpq_gcd(numerator.coeffs() + denominator.coeffs())
            gcd *= content
        self.numerator = numerator / gcd
        self.denominator = denominator / gcd
        self.ctx = ctx

    @staticmethod
    def from_sympy(
        rational: sp.Expr,
        ctx: FlintContext,
        convert=None,
    ) -> FlintRational:
        r"""
        Converts a rational function given as a sympy expression to a FlintRational.
        Args:
            rational: The expression to convert to flint
            ctx: The desired mpoly context (which also defines the supported variables)
        Returns:
            A FlintRational object representing the `rational` value
        """
        convert = convert or flint_converter(ctx)
        numerator, denominator = sp.fraction(rational, exact=True)
        try:
            numerator, denominator = convert(numerator), convert(denominator)
        except (sp.PolynomialError, TypeError):
            numerator, denominator = map(convert, rational.as_numer_denom())
        return FlintRational(numerator, denominator, ctx)

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
            left = _PolynomialFraction(self.numerator, self.denominator)
            right = _PolynomialFraction(other.numerator, other.denominator)
            product = left * right
            return FlintRational(product.numerator, product.denominator, self.ctx)
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
        return self.numerator * other.denominator == self.denominator * other.numerator

    def degrees(self) -> list[int]:
        return [max(poly.degrees()) for poly in [self.numerator, self.denominator]]

    def subs(self, substitutions: dict) -> FlintRational:
        """
        Substitutes symbols in self.
        """
        substitutions = Position(substitutions)
        composition = _flint_composition(self.ctx, substitutions)
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
        return self.to_sympy()

    def to_sympy(self, factor: bool = True) -> sp.Expr:
        """Convert to SymPy, optionally without irreducible factorization."""
        return flint_to_sympy(self.numerator, factor=factor) / flint_to_sympy(
            self.denominator,
            factor=factor,
        )
