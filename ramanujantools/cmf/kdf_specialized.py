"""
Exact argument-specialized construction of KdF CMFs, used when the generic basis degenerates (e.g. at x=y=1).

U and V represent the ordinary derivatives of the KdF function in x and y. They shift the parameters
(a, b; d, e) and (a, c; d, f) respectively by +1 in physical coordinates, so they do not commute with the coefficients.
A completed Groebner basis of the relations in this two-shift Ore algebra gives a basis of the *contiguity module*
of the function at the given arguments. This is not necessarily the full solution space of the Horn system,
and its rank can differ from the generic rank.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import sympy as sp
from sympy.abc import x, y
from sympy.functions.combinatorial.numbers import stirling

from ramanujantools.flint_core import FlintRational
from ramanujantools.flint_core.context import FlintContext, flint_ctx


class FlintFraction:
    """A rational function with python-flint multivariate polynomials as numerator and denominator."""

    def __init__(self, field, numerator, denominator=None):
        self.field = field
        denominator = field.ctx.constant(1) if denominator is None else denominator
        if not denominator:
            raise ZeroDivisionError("zero rational-function denominator")
        common = numerator.gcd(denominator)
        self.numerator, self.denominator = numerator / common, denominator / common

    def __bool__(self):
        return bool(self.numerator)

    def coerce(self, value):
        if isinstance(value, FlintFraction):
            return value
        return self.field.from_sympy(sp.sympify(value))

    def __neg__(self):
        return FlintFraction(self.field, -self.numerator, self.denominator)

    def __add__(self, other):
        other = self.coerce(other)
        common = self.denominator.gcd(other.denominator)
        left, right = self.denominator / common, other.denominator / common
        return FlintFraction(
            self.field,
            self.numerator * right + other.numerator * left,
            left * other.denominator,
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        a = self.numerator.gcd(other.denominator)
        b = other.numerator.gcd(self.denominator)
        return FlintFraction(
            self.field,
            (self.numerator / a) * (other.numerator / b),
            (self.denominator / b) * (other.denominator / a),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        return self * FlintFraction(self.field, other.denominator, other.numerator)

    def translate(self, shifts):
        if not any(shifts):
            return self
        values = [g + n for g, n in zip(self.field.ctx.gens(), shifts)]
        return FlintFraction(
            self.field,
            self.numerator.compose(*values),
            self.denominator.compose(*values),
        )


class FlintField:
    def __init__(self, symbols):
        self.symbols = tuple(sorted(symbols, key=str))
        self.ctx = flint_ctx(self.symbols, fmpz=False)
        self.zero = FlintFraction(self, self.ctx.constant(0))
        self.one = FlintFraction(self, self.ctx.constant(1))

    def from_sympy(self, expression):
        import flint

        def convert(p):
            return self.ctx.from_dict(
                {
                    power: flint.fmpq(int(c.p), int(c.q))
                    for power, c in sp.Poly(p, *self.symbols).terms()
                }
            )

        numerator, denominator = expression.as_numer_denom()
        return FlintFraction(self, convert(numerator), convert(denominator))


def order(power):
    return sum(power), power[0], power[1]


class ShiftAlgebra:
    """Sparse left Groebner reduction with commuting translation automorphisms."""

    def __init__(self, symbols, shifts):
        self.shifts = shifts
        self.field = FlintField(symbols)
        self.basis = []
        self.pairs_checked = 0

    @lru_cache(maxsize=4096)
    def shift(self, coefficient, power):
        return coefficient.translate(
            tuple(power[0] * u + power[1] * v for u, v in self.shifts)
        )

    def parameter_shift(self, coefficient, index):
        return coefficient.translate(
            tuple(int(i == index) for i in range(len(self.shifts)))
        )

    def left(self, polynomial, power):
        return {
            (i + power[0], j + power[1]): self.shift(c, power)
            for (i, j), c in polynomial.items()
            if c
        }

    @staticmethod
    def subtract(target, source, scale):
        for power, coefficient in source.items():
            value = target.get(power, 0) - scale * coefficient
            if value:
                target[power] = value
            else:
                target.pop(power, None)

    def reduce(self, polynomial, basis=None):
        basis = self.basis if basis is None else basis
        reducers = [(max(g, key=order), g) for g in basis]
        current, remainder = dict(polynomial), {}
        while current:
            power = max(current, key=order)
            for lead, g in reducers:
                delta = (power[0] - lead[0], power[1] - lead[1])
                if min(delta) >= 0:
                    multiple = self.left(g, delta)
                    self.subtract(current, multiple, current[power] / multiple[power])
                    break
            else:
                remainder[power] = current.pop(power)
        return remainder

    @staticmethod
    def monic(polynomial):
        coefficient = polynomial[max(polynomial, key=order)]
        return {p: c / coefficient for p, c in polynomial.items()}

    def complete(self, relations, max_pairs):
        pending = []

        def insert(relation):
            remainder = self.reduce(relation)
            if remainder:
                index = len(self.basis)
                self.basis.append(self.monic(remainder))
                pending.extend((i, index) for i in range(index))

        for relation in sorted(
            (g for g in relations if g), key=lambda g: order(max(g, key=order))
        ):
            insert(relation)
        while pending:
            if self.pairs_checked >= max_pairs:
                raise ValueError(
                    f"Shift Groebner completion exceeded max_pairs={max_pairs}; no CMF was certified"
                )

            def pair_order(pair):
                a, b = (max(self.basis[i], key=order) for i in pair)
                return order((max(a[0], b[0]), max(a[1], b[1])))

            pair = min(pending, key=pair_order)
            pending.remove(pair)
            f, g = (self.basis[i] for i in pair)
            a, b = max(f, key=order), max(g, key=order)
            common = max(a[0], b[0]), max(a[1], b[1])
            s = self.left(f, (common[0] - a[0], common[1] - a[1]))
            self.subtract(
                s,
                self.left(g, (common[0] - b[0], common[1] - b[1])),
                self.field.one,
            )
            self.pairs_checked += 1
            insert(s)
        # Remove redundant leading monomials, then reduce every tail.
        minimal = []
        for g in sorted(self.basis, key=lambda g: order(max(g, key=order))):
            lead = max(g, key=order)
            if not any(
                all(a <= b for a, b in zip(max(h, key=order), lead)) for h in minimal
            ):
                minimal.append(g)
        self.basis = [
            self.monic(self.reduce(g, minimal[:i] + minimal[i + 1 :]))
            for i, g in enumerate(minimal)
        ]
        if any(self.reduce(g) for g in relations):
            raise ValueError("An input relation failed the completed reduction")
        return self.standard_monomials()

    def standard_monomials(self):
        leads = [max(g, key=order) for g in self.basis]
        if (0, 0) in leads:
            raise ValueError("The specialized relations give the zero module")
        horizontal = [i for i, j in leads if j == 0]
        vertical = [j for i, j in leads if i == 0]
        if not horizontal or not vertical:
            raise ValueError(
                "The specialized shift module is not finite-dimensional; no finite CMF was established"
            )
        return tuple(
            sorted(
                (
                    (i, j)
                    for i in range(min(horizontal))
                    for j in range(min(vertical))
                    if not any(a <= i and b <= j for a, b in leads)
                ),
                key=order,
            )
        )


def construct(
    operators: tuple,
    x_value: sp.Expr,
    y_value: sp.Expr,
    blocks: tuple[tuple[sp.Symbol, ...], ...],
    max_pairs: int,
) -> tuple[tuple[tuple[int, int], ...], list, list, FlintContext]:
    r"""
    Returns the powers $(i, j)$ of the basis of ordinary derivatives $\partial_x^i \partial_y^j$,
    the matrices of $\theta_x, \theta_y$ in that basis (as rows of `FlintRational`) and their FLINT context.

    Since reduction to normal form is linear, the matrix of an axis is $I + (h \theta_x + v \theta_y) / N$,
    where $N$ is the normalizer of the axis and $h, v$ indicate whether it is shifted by $\partial_x, \partial_y$.
    Every axis is verified to transport the relations of the module.

    Args:
        operators: the two Horn operators and the elimination operator (polynomials in theta_x, theta_y, x, y).
        blocks: the parameter blocks (a, b, c, nd, ne, nf).
    """
    from ramanujantools.cmf.kdf import theta_x, theta_y

    a, b, c, d, e, f = blocks
    parameters = sum(blocks, ())
    symbols = sorted(
        set(parameters) | x_value.free_symbols | y_value.free_symbols, key=str
    )
    vx = dict.fromkeys(a + b, 1) | dict.fromkeys(d + e, -1)
    vy = dict.fromkeys(a + c, 1) | dict.fromkeys(d + f, -1)
    algebra = ShiftAlgebra(
        symbols, tuple((vx.get(s, 0), vy.get(s, 0)) for s in symbols)
    )
    field = algebra.field

    # theta^n = sum_k S(n,k) z^k partial_z^k. The conversion is done BEFORE evaluating the arguments.
    relations = []
    for h in operators:
        terms = {}
        for (i, j), coefficient in sp.Poly(h, theta_x, theta_y).terms():
            for u in range(i + 1):
                for v in range(j + 1):
                    value = (
                        coefficient
                        * stirling(i, u, kind=2)
                        * stirling(j, v, kind=2)
                        * x**u
                        * y**v
                    )
                    if value:
                        terms[u, v] = terms.get((u, v), 0) + value
        terms = {p: sp.expand(c) for p, c in terms.items() if sp.expand(c) != 0}
        # Remove common argument monomials before evaluation (valid for the analytic KdF family, also at x=0 or y=0).
        argument_powers = [
            power for c in terms.values() for power, _ in sp.Poly(c, x, y).terms()
        ]
        divisor = x ** min((p[0] for p in argument_powers), default=0) * y ** min(
            (p[1] for p in argument_powers), default=0
        )
        relation = {
            p: field.from_sympy(
                sp.cancel(c / divisor).subs({x: x_value, y: y_value}, simultaneous=True)
            )
            for p, c in terms.items()
        }
        relations.append({p: c for p, c in relation.items() if c})
    powers = algebra.complete(relations, max_pairs=max_pairs)

    normal_forms = {}

    def normal_form(power) -> list:
        if power not in normal_forms:
            reduced = algebra.reduce({power: field.one})
            normal_forms[power] = [reduced.get(p, field.zero) for p in powers]
        return normal_forms[power]

    arguments = (field.from_sympy(x_value), field.from_sympy(y_value))

    @lru_cache
    def theta_image(power, direction) -> list:
        r"""
        The normal form of $\partial^p \theta F$ for $\theta=\theta_x$ (direction 0) or $\theta_y$ (direction 1):
        $\partial_x^i \theta_x = (\theta_x + i) \partial_x^i$, so it is $i \partial^p + x \partial^{p+(1,0)}$.
        """
        shifted = list(power)
        shifted[direction] += 1
        return [
            power[direction] * value + arguments[direction] * shifted_value
            for value, shifted_value in zip(
                normal_form(power), normal_form(tuple(shifted))
            )
        ]

    theta_matrices = [
        list(zip(*(theta_image(p, direction) for p in powers))) for direction in (0, 1)
    ]

    # The matrix of an axis maps the target module (at lambda + e_axis) to the source module (at lambda).
    # It is well defined only if it maps every relation of the target module to zero, which is verified here.
    # The relations are symmetric within each block (the reduced Groebner basis is unique),
    # so it suffices to verify the first axis of every block.
    lower_axes = set(d + e + f)
    for axis in (block[0] for block in blocks if block):
        normalizer = field.from_sympy(-axis - 1 if axis in lower_axes else axis)
        directions = [direction for direction, v in enumerate((vx, vy)) if axis in v]
        index = symbols.index(axis)
        for relation in algebra.basis:
            total = [field.zero] * len(powers)
            for power, coefficient in relation.items():
                coefficient = algebra.parameter_shift(coefficient, index)
                image = [normalizer * value for value in normal_form(power)]
                for direction in directions:
                    image = [
                        u + v for u, v in zip(image, theta_image(power, direction))
                    ]
                total = [t + coefficient * v for t, v in zip(total, image)]
            if any(total):
                raise ValueError(f"Specialized relation transport failed for {axis}")

    # Exact intertwining and commuting contiguity operators prove conservation (by uniqueness of normal forms).
    contiguity = {}
    for axis in parameters:
        theta = (
            theta_x + theta_y
            if axis in a + d
            else theta_x if axis in b + e else theta_y
        )
        normalizer = -axis - 1 if axis in lower_axes else axis
        contiguity[axis] = (normalizer + theta) / normalizer
    for u, v in combinations(parameters, 2):
        residual = contiguity[u] * contiguity[v].subs(u, u + 1) - contiguity[
            v
        ] * contiguity[u].subs(v, v + 1)
        if sp.cancel(residual) != 0:
            raise ValueError(f"Contiguity operator commutation failed for {u}, {v}")

    ctx = field.ctx
    theta_x_matrix, theta_y_matrix = (
        [[FlintRational(v.numerator, v.denominator, ctx) for v in row] for row in m]
        for m in theta_matrices
    )
    return powers, theta_x_matrix, theta_y_matrix, ctx
