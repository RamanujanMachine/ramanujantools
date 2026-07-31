from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, reduce

import flint
import sympy as sp

import ramanujantools as rt
from ramanujantools import Position
from ramanujantools.flint_core.context import (
    FlintContext,
    flint_converter,
    flint_to_sympy,
    _flint_composition,
    _fmpz_mpoly_to_fmpz_poly,
)
from ramanujantools.flint_core.rational import (
    FlintRational,
    _PolynomialFraction,
    _polynomial_gcd,
    _polynomial_lcm,
)
from ramanujantools.utils import batched, Batchable


class SymbolicMatrix:
    """
    Represents a rational-function matrix as one scalar times a primitive
    polynomial matrix.

    It's logic is limited compared to the main Matrix, as it's designed for bottlenecks.
    """

    def __init__(
        self, rows: int, cols: int, values: list[FlintRational], ctx: FlintContext
    ) -> SymbolicMatrix:
        self._rows = rows
        self._cols = cols
        self.ctx = ctx
        self._set_from_rationals(values)

    @classmethod
    def _from_polynomials(
        cls,
        rows: int,
        cols: int,
        polynomials: list,
        scalar: FlintRational,
        normalize: bool = False,
    ) -> SymbolicMatrix:
        matrix = object.__new__(cls)
        matrix._rows = rows
        matrix._cols = cols
        matrix.ctx = scalar.ctx
        matrix.polynomials = list(polynomials)
        matrix.scalar = scalar
        if matrix.scalar.denominator == 0:
            raise ZeroDivisionError("Symbolic matrix has a zero scalar denominator")
        if normalize:
            matrix._normalize()
        return matrix

    def _normalize(self) -> None:
        content = self.ctx.constant(0)
        for polynomial in self.polynomials:
            content = content.gcd(polynomial)
            if content == 1:
                return
        if content != 0:
            self.polynomials = [polynomial / content for polynomial in self.polynomials]
            self.scalar *= content

    def _set_from_rationals(self, values: list[FlintRational]) -> None:
        if len(values) != self.rows() * self.cols():
            raise ValueError(
                f"Expected {self.rows() * self.cols()} values, got {len(values)}"
            )
        if not values:
            self.polynomials = []
            self.scalar = FlintRational(
                self.ctx.constant(1),
                self.ctx.constant(1),
                self.ctx,
            )
            return

        denominator = reduce(_polynomial_lcm, (value.denominator for value in values))
        self.polynomials = [
            value.numerator * (denominator / value.denominator) for value in values
        ]
        self.scalar = FlintRational(
            self.ctx.constant(1),
            denominator,
            self.ctx,
        )
        self._normalize()

    @staticmethod
    def from_sympy(matrix: rt.Matrix, ctx: FlintContext) -> SymbolicMatrix:
        """
        Converts a Matrix to SymbolicMatrix.
        Args:
            matrix: The matrix as ramanujantools.Matrix
            ctx: The desired mpoly context (which also defines the supported variables)
        """
        convert = flint_converter(ctx)
        values = [FlintRational.from_sympy(cell, ctx, convert) for cell in matrix]
        return SymbolicMatrix(matrix.rows, matrix.cols, values, ctx)

    @staticmethod
    def eye(N: int, ctx: FlintContext) -> SymbolicMatrix:
        """
        Creates an identity matrix of size N.

        Args:
            N: The squared matrix dimension
            ctx: The desired mpoly context (which also defines the supported variables)
        """
        one = ctx.constant(1)
        zero = ctx.constant(0)
        polynomials = [
            one if row == column else zero for row in range(N) for column in range(N)
        ]
        return SymbolicMatrix._from_polynomials(
            N,
            N,
            polynomials,
            FlintRational(one, one, ctx),
        )

    @property
    def values(self) -> list[FlintRational]:
        """Materialize the historical per-entry rational representation."""
        return [self.scalar * polynomial for polynomial in self.polynomials]

    def __getitem__(self, key):
        """
        Returns an element of the matrix.
        Supports both matrix[row, col] and matrix[index] syntax
        """
        if isinstance(key, tuple):
            row, col = key
            return self.scalar * self.polynomials[row * self.cols() + col]
        return self.scalar * self.polynomials[key]

    def __setitem__(self, key, value):
        """
        Returns an element of the matrix.
        Supports both matrix[row, col] and matrix[index] syntax
        """
        values = self.values
        index = key[0] * self.cols() + key[1] if isinstance(key, tuple) else key
        values[index] = value
        self._set_from_rationals(values)

    def __eq__(self, other: SymbolicMatrix):
        return (
            isinstance(other, SymbolicMatrix)
            and self.shape() == other.shape()
            and self.values == other.values
        )

    def rows(self):
        return self._rows

    def cols(self):
        return self._cols

    def shape(self):
        return (self.rows(), self.cols())

    def row(self, index: int) -> list[FlintRational]:
        return [self[index, column] for column in range(self.cols())]

    def col(self, index: int) -> list[FlintRational]:
        return [self[row, index] for row in range(self.rows())]

    def data(self) -> list[list[FlintRational]]:
        return [self.row(row) for row in range(self.rows())]

    def __repr__(self) -> str:
        return f"SymbolicMatrix({self.data()})"

    __str__ = __repr__

    def __mul__(self, other: SymbolicMatrix | int) -> SymbolicMatrix:
        """
        Multiplies self by another SymbolicMatrix or a scalar.
        """
        if isinstance(other, SymbolicMatrix):
            if self.cols() != other.rows():
                raise ValueError(
                    "Attempting to multiply matrices with incompatible shapes!"
                    f"self.cols()={self.cols()}, other.rows()={other.rows()}"
                )
            elements = [
                sum(
                    (
                        self.polynomials[row * self.cols() + k]
                        * other.polynomials[k * other.cols() + col]
                        for k in range(self.cols())
                    ),
                    self.ctx.constant(0),
                )
                for row in range(self.rows())
                for col in range(other.cols())
            ]
            return SymbolicMatrix._from_polynomials(
                self.rows(),
                other.cols(),
                elements,
                self.scalar * other.scalar,
                normalize=True,
            )

        return SymbolicMatrix._from_polynomials(
            self.rows(),
            self.cols(),
            self.polynomials,
            self.scalar * other,
        )

    def __rmul__(self, other: SymbolicMatrix | int) -> SymbolicMatrix:
        """
        Multiplies a SymbolicMatrix or a scalar by self.
        """
        if isinstance(other, int):
            return self * other
        return other * self

    def __truediv__(self, other: int) -> SymbolicMatrix:
        """
        Divides self by a scalar
        """
        if isinstance(other, SymbolicMatrix):
            raise ValueError("Attempted to divide by matrix!")
        return SymbolicMatrix._from_polynomials(
            self.rows(),
            self.cols(),
            self.polynomials,
            self.scalar / other,
        )

    def subs(self, substitutions: dict) -> SymbolicMatrix:
        """
        Substitutes symbols in the matrix.
        """
        return self._subs(substitutions, normalize=True)

    def shift(self, symbol: sp.Symbol, amount: int) -> SymbolicMatrix:
        """Apply an invertible integer shift without recomputing matrix content."""
        return self._subs({symbol: symbol + amount}, normalize=False)

    def inverse(self) -> SymbolicMatrix:
        """Return the exact inverse using fraction-free FLINT elimination."""
        if self.rows() != self.cols():
            raise ValueError("Only square matrices can be inverted")
        size = self.rows()
        if size == 0:
            return SymbolicMatrix._from_polynomials(
                0, 0, [], self.scalar.inv(), normalize=False
            )

        coefficients = [
            self.polynomials[row * size : (row + 1) * size]
            for row in range(size)
        ]
        numerators, denominator = _bareiss_inverse(coefficients)
        polynomials = [
            numerators[row][column]
            for row in range(size)
            for column in range(size)
        ]
        scalar = self.scalar.inv() / denominator
        return SymbolicMatrix._from_polynomials(
            size, size, polynomials, scalar, normalize=True
        )

    def _subs(self, substitutions: dict, normalize: bool) -> SymbolicMatrix:
        composition = _flint_composition(self.ctx, substitutions)
        return SymbolicMatrix._from_polynomials(
            self.rows(),
            self.cols(),
            [polynomial.compose(*composition) for polynomial in self.polynomials],
            self.scalar.subs(substitutions),
            normalize=normalize,
        )

    def factor(self) -> rt.Matrix:
        """
        Factors all elements in the matrix.
        """
        return self.to_rt(factor=True)

    def to_rt(self, factor: bool = False) -> rt.Matrix:
        """Convert to a ramanujantools Matrix, optionally factoring entries."""
        scalar = self.scalar.to_sympy(factor=factor)
        polynomials = [
            flint_to_sympy(polynomial, factor=factor) for polynomial in self.polynomials
        ]
        if scalar == 0:
            values = [sp.S.Zero] * len(polynomials)
        elif scalar == 1:
            values = polynomials
        elif factor:
            values = [scalar * polynomial for polynomial in polynomials]
        else:
            values = [
                sp.S.Zero
                if polynomial == 0
                else (
                    scalar
                    if polynomial == 1
                    else sp.Mul(scalar, polynomial, evaluate=False)
                )
                for polynomial in polynomials
            ]
        return rt.Matrix(self.rows(), self.cols(), values)

    @batched("iterations")
    def walk(
        self, trajectory: dict, iterations: Batchable[int], start: dict
    ) -> Batchable[SymbolicMatrix]:
        r"""
        Returns the multiplication result of walking in a certain trajectory.

        The `walk` operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=self`, `(t_0, ..., t_k)=trajectory`, `n=iterations` and `(s_0, ..., s_k)=start`.

        This is a generalization of the basic (and most common) case $\prod_{i=0}^{n-1}M(s+i)$,
        where M=self, n=iterations and s=start.

        Args:
            trajectory: the trajectory of a single step in the walk, as defined above.
            iterations: The amount of multiplications to perform. Can be an integer value or a list of values.
            start: the starting point of the matrix multiplication
        Returns:
            The walk multiplication matrix as defined above.
            If iterations is list, returns a list of matrices.
        Raises:
            ValueError: If `self` is not a square matrix,
                        if `start` and `trajectory` have different keys,
                        if `iterations` contains duplicate values
        """
        position = Position(start)
        trajectory = Position(trajectory)
        results = []
        matrix = SymbolicMatrix.eye(self.rows(), self.ctx)
        for depth in range(0, iterations[-1]):
            if depth in iterations:
                results.append(matrix)
            matrix *= self.subs(position)
            position += trajectory
        results.append(matrix)  # Last matrix, for iterations[-1]
        return results

    def companionize(self, symbol: sp.Symbol) -> _CompanionData:
        """Find and certify the minimal shifted-Krylov dependency."""
        dimension = self.rows()
        one = self.ctx.constant(1)
        zero = self.ctx.constant(0)
        columns = [
            SymbolicMatrix._from_polynomials(
                dimension,
                1,
                [one, *([zero] * (dimension - 1))],
                FlintRational(one, one, self.ctx),
            )
        ]
        univariate = len(self.ctx.gens()) == 1
        convert = (
            _fmpz_mpoly_to_fmpz_poly if univariate else lambda polynomial: polynomial
        )
        pivot_rows = (0,)
        for rank in range(1, dimension + 1):
            columns.append(self * columns[-1].shift(symbol, 1))
            dependency, witness_row = _primitive_dependency(
                columns, pivot_rows, convert
            )
            if dependency is None:
                if witness_row is None:
                    raise ArithmeticError(
                        "Independent Krylov column has no nonzero residual"
                    )
                pivot_rows += (witness_row,)
                continue
            scalar_shifts = (
                tuple(
                    convert(
                        self.scalar.numerator.compose(
                            *_flint_composition(self.ctx, {symbol: symbol + shift})
                        )
                    )
                    for shift in range(rank)
                )
                if univariate and self.scalar.numerator != 1
                else ()
            )
            coefficients = _original_coefficients(
                dependency, columns, convert, scalar_shifts
            )
            return _CompanionData(self, symbol, columns, coefficients, scalar_shifts)
        raise ValueError("Could not find a shifted-Krylov companion relation")


def _bareiss_inverse(coefficients) -> tuple[list[list], object]:
    """Invert a polynomial matrix with one fraction-free elimination."""
    size = len(coefficients)
    if size == 0:
        raise ValueError("Cannot invert an empty matrix")
    one = coefficients[0][0] ** 0
    zero = one - one
    values = [
        [*row, *(one if index == column else zero for column in range(size))]
        for index, row in enumerate(coefficients)
    ]

    previous_pivot = one
    for k in range(size - 1):
        pivot_row = next(
            (row for row in range(k, size) if values[row][k] != 0),
            None,
        )
        if pivot_row is None:
            raise ZeroDivisionError("Singular polynomial matrix")
        if pivot_row != k:
            values[k], values[pivot_row] = values[pivot_row], values[k]

        pivot = values[k][k]
        for row in range(k + 1, size):
            multiplier = values[row][k]
            for column in range(k + 1, 2 * size):
                numerator = pivot * values[row][column] - multiplier * values[k][column]
                values[row][column], remainder = divmod(numerator, previous_pivot)
                if remainder != 0:
                    raise ArithmeticError("Non-exact Bareiss elimination")
            values[row][k] = zero
        previous_pivot = pivot

    denominator = values[-1][size - 1]
    if denominator == 0:
        raise ZeroDivisionError("Singular polynomial matrix")
    result = [[zero] * size for _ in range(size)]
    for rhs in range(size):
        column = [zero] * size
        for row in reversed(range(size)):
            value = values[row][size + rhs] * denominator
            for index in range(row + 1, size):
                value -= values[row][index] * column[index]
            column[row], remainder = divmod(value, values[row][row])
            if remainder != 0:
                raise ArithmeticError("Non-exact Bareiss back-substitution")
        for row in range(size):
            result[row][rhs] = column[row]
    return result, denominator


def _bareiss_solve(coefficients, rhs) -> tuple[list, object]:
    """Solve a square polynomial system by fraction-free elimination."""
    size = len(rhs)
    values = [[*row, value] for row, value in zip(coefficients, rhs)]
    one = values[0][0] ** 0
    zero = one - one

    previous_pivot = one
    for k in range(size - 1):
        pivot_row = next(
            (i for i in range(k, size) if values[i][k] != 0),
            None,
        )
        if pivot_row is None:
            raise ZeroDivisionError("Singular polynomial system")
        if pivot_row != k:
            values[k], values[pivot_row] = values[pivot_row], values[k]

        pivot = values[k][k]
        for i in range(k + 1, size):
            multiplier = values[i][k]
            for j in range(k + 1, size + 1):
                values[i][j] = (
                    pivot * values[i][j] - multiplier * values[k][j]
                ) / previous_pivot
            values[i][k] = zero
        previous_pivot = pivot

    denominator = values[-1][-2]
    if denominator == 0:
        raise ZeroDivisionError("Singular polynomial system")
    numerators = [zero] * size
    for i in reversed(range(size)):
        value = values[i][-1] * denominator
        for j in range(i + 1, size):
            value -= values[i][j] * numerators[j]
        numerators[i], remainder = divmod(value, values[i][i])
        if remainder != 0:
            raise ArithmeticError("Non-exact Bareiss back-substitution")

    return numerators, denominator


def _divide_factors(polynomial, factors):
    for factor in factors:
        polynomial, remainder = divmod(polynomial, factor)
        if remainder != 0:
            return None
    return polynomial


def _primitive_dependency(
    columns: list[SymbolicMatrix],
    pivot_rows: tuple[int, ...],
    convert,
) -> tuple[tuple[tuple, object] | None, int | None]:
    """Return an exact dependency or a residual row certifying independence."""
    basis, target = columns[:-1], columns[-1]
    rank = len(basis)
    primitive_columns = [
        [convert(polynomial) for polynomial in column.polynomials] for column in columns
    ]
    *primitive_basis, primitive_target = primitive_columns
    one = primitive_basis[0][0]
    zero = one - one

    if rank == 1:
        numerators = [primitive_target[0]]
        denominator = one
    else:
        coefficients = [
            [primitive_basis[column][row] for column in range(1, rank)]
            for row in pivot_rows[1:]
        ]
        rhs = [primitive_target[row] for row in pivot_rows[1:]]
        tail, denominator = _bareiss_solve(coefficients, rhs)
        numerators = [
            primitive_target[0] * denominator
            - sum(
                (
                    primitive_basis[column][0] * numerator
                    for column, numerator in enumerate(tail, start=1)
                ),
                zero,
            ),
            *tail,
        ]

    common = reduce(_polynomial_gcd, [denominator, *numerators])
    numerators = [numerator / common for numerator in numerators]
    denominator /= common
    for row in range(target.rows()):
        if row in pivot_rows:
            continue
        left = sum(
            (
                primitive_basis[column][row] * numerator
                for column, numerator in enumerate(numerators)
            ),
            zero,
        )
        if left != primitive_target[row] * denominator:
            return None, row
    return (tuple(numerators), denominator), None


def _original_coefficients(
    primitive_dependency: tuple[tuple, object],
    columns: list[SymbolicMatrix],
    convert,
    scalar_shifts: tuple[flint.fmpz_poly, ...],
) -> tuple[_PolynomialFraction, ...]:
    basis, target = columns[:-1], columns[-1]
    primitive_numerators, primitive_denominator = primitive_dependency
    target_numerator = convert(target.scalar.numerator)
    target_denominator = convert(target.scalar.denominator)
    target_scalar = _PolynomialFraction(target_numerator, target_denominator)
    target_residual = _divide_factors(target_numerator, scalar_shifts)
    result = []
    for index, (primitive_numerator, column) in enumerate(
        zip(primitive_numerators, basis)
    ):
        primitive = _PolynomialFraction(primitive_numerator, primitive_denominator)
        column_numerator = convert(column.scalar.numerator)
        column_denominator = convert(column.scalar.denominator)
        column_residual = _divide_factors(column_numerator, scalar_shifts[:index])
        if target_residual is not None and column_residual is not None:
            reduced = primitive * (
                _PolynomialFraction(target_residual, target_denominator)
                * _PolynomialFraction(column_denominator, column_residual)
            )
            coefficient = _PolynomialFraction(
                reduced.numerator,
                reduced.denominator,
                tuple(range(index, len(basis))) if scalar_shifts else (),
            )
        else:
            coefficient = (
                primitive
                * target_scalar
                * _PolynomialFraction(column_denominator, column_numerator)
            )
        result.append(coefficient)
    return tuple(result)


@dataclass
class _CompanionData:
    """Cached FLINT data backing all public companion projections."""

    matrix: SymbolicMatrix
    symbol: sp.Symbol
    columns: list[SymbolicMatrix]
    coefficients: tuple[_PolynomialFraction, ...]
    scalar_shifts: tuple[flint.fmpz_poly, ...]

    @property
    def rank(self) -> int:
        return len(self.coefficients)

    def _to_sympy(self, polynomial) -> sp.Expr:
        return flint_to_sympy(
            polynomial,
            factor=False,
            symbol=self.symbol,
        )

    def _coefficient_numerator_to_sympy(
        self,
        value: _PolynomialFraction,
    ) -> sp.Expr:
        factors = [self._to_sympy(value.numerator)]
        factors.extend(
            self._to_sympy(self.scalar_shifts[shift]) for shift in value.scalar_shifts
        )
        return sp.Mul(
            *(factor for factor in factors if factor != 1),
            evaluate=False,
        )

    def _coefficient_to_sympy(self, value: _PolynomialFraction) -> sp.Expr:
        return self._coefficient_numerator_to_sympy(value) / self._to_sympy(
            value.denominator
        )

    @cached_property
    def companion(self):
        return rt.Matrix.companion_form(
            [self._coefficient_to_sympy(value) for value in self.coefficients]
        )

    @cached_property
    def coboundary(self):
        while len(self.columns) < self.matrix.rows():
            self.columns.append(self.matrix * self.columns[-1].shift(self.symbol, 1))
        return rt.Matrix.hstack(
            *(
                column.to_rt(factor=False)
                for column in self.columns[: self.matrix.rows()]
            )
        )

    @cached_property
    def recurrence(self) -> list[sp.Expr]:
        denominator = reduce(
            _polynomial_lcm,
            (coefficient.denominator for coefficient in self.coefficients),
        )
        relation = [-self._to_sympy(denominator)]
        for coefficient in reversed(self.coefficients):
            numerator = self._coefficient_numerator_to_sympy(coefficient)
            multiplier = self._to_sympy(denominator / coefficient.denominator)
            relation.append(
                numerator
                if multiplier == 1
                else sp.Mul(multiplier, numerator, evaluate=False)
            )
        return relation
