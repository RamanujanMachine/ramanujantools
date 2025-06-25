from __future__ import annotations

import mpmath as mp
import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix, Limit, LinearRecurrence
from ramanujantools.utils import batched, Batchable


def is_deflatable(a_factors, b_factors, factor):
    if n in factor.free_symbols:
        return (
            a_factors.get(factor, 0) > 0
            and b_factors.get(factor, 0) > 0
            and b_factors.get(factor.subs({n: n - 1}).expand(), 0) > 0
        )
    else:
        return a_factors.get(factor, 0) > 0 and b_factors.get(factor, 0) > 1


def remove_factor(a_factors, b_factors, factor):
    a_factors[factor] -= 1
    b_factors[factor] -= 1
    b_factors[factor.subs({n: n - 1}).expand()] -= 1


def deflate_constant(a_constant, b_constant):
    factors = sp.factorint(sp.gcd(a_constant**2, b_constant))
    constant = 1
    for root, mul in factors.items():
        constant *= root ** (mul // 2)
    return constant


def content(a, b, variables):
    if len(a.free_symbols | b.free_symbols) == 0:
        return deflate_constant(a, b)

    def factor_list(poly, variables):
        content, factors = sp.factor_list(poly, *variables)
        return content, {
            factor.expand(): power for factor, power in dict(factors).items()
        }

    (a_content, a_factors), (b_content, b_factors) = map(
        lambda p: factor_list(p, variables), [a, b]
    )

    c_n = content(a_content, b_content, [])
    for factor in a_factors:
        while is_deflatable(a_factors, b_factors, factor):
            remove_factor(a_factors, b_factors, factor)
            c_n *= factor

    return sp.simplify(c_n)


class PCF:
    """
    Represents a Polynomial Continued Fraction (PCF).
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a PCF. Accepts one of:
        - a_n: sp.Expr, b_n: sp.Expr
        - recurrence: LinearRecurrence
        - matrix: Matrix.
        """
        if len(args) == 2 and all(isinstance(arg, (int, sp.Expr)) for arg in args):
            a_n, b_n = args
            if b_n == 0:
                raise ValueError("b_n cannot be zero in a PCF!")
            self.recurrence = LinearRecurrence([1, a_n, b_n])
        elif len(args) == 1 and isinstance(recurrence := args[0], LinearRecurrence):
            self.recurrence = recurrence.normalize()
        elif len(args) == 1 and isinstance(matrix := args[0], Matrix):
            if not matrix.is_square() or matrix.rows != 2:
                raise ValueError(
                    "Only a 2x2 matrix can be converted to a PCF, got {obj.rows}x{obj.cols}"
                )
            self.recurrence = LinearRecurrence(matrix).normalize()
        else:
            raise ValueError(
                f"Invalid PCF arguments: got {', '.join(type(arg).__name__ for arg in args)}"
            )

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"PCF({self.a_n}, {self.b_n})"

    def __eq__(self, other: PCF) -> bool:
        return self.recurrence == other.recurrence

    @property
    def a_n(self) -> sp.Expr:
        return self.recurrence.relation[1]

    @property
    def b_n(self) -> sp.Expr:
        return self.recurrence.relation[2]

    def _latex(self, printer) -> str:
        DEPTH = 3

        def recurse(i: int, depth: int) -> str:
            if i == depth:
                b_n_latex = printer.doprint(self.b_n)
                a_n_latex = printer.doprint(self.a_n)
                return rf"\ddots + \cfrac{{{b_n_latex}}}{{{a_n_latex}}}"
            else:
                a_i = printer.doprint(self.a_n.subs({n: i}))
                b_i = printer.doprint(self.b_n.subs({n: i + 1}))
                return rf"{a_i} + \cfrac{{{b_i}}}{{{recurse(i + 1, DEPTH)}}}"

        a_0 = printer.doprint(self.a_n.subs({n: 0}))
        b_1 = printer.doprint(self.b_n.subs({n: 1}))
        return rf"{a_0} + \cfrac{{{b_1}}}{{{recurse(1, DEPTH)}}}"

    def _repr_latex_(self) -> str:
        return rf"$${sp.latex(self)}$$"

    def degrees(self) -> tuple[int, int]:
        """
        Returns the degrees of a_n and b_n as a tuple: $(deg(a_n), deg(b_n))$
        """
        return tuple(map(lambda p: sp.Poly(p, n).degree(), [self.a_n, self.b_n]))

    def M(self) -> Matrix:
        r"""
        Returns the matrix that represents the PCF recurrence:

        $M = \begin{pmatrix} 0, b_n \cr 1, a_n \end{pmatrix}$
        """
        return self.recurrence.recurrence_matrix

    def A(self) -> Matrix:
        r"""
        Returns the matrix that represents the $a_0$ part:

        $A = \begin{pmatrix} 1, a_0 \cr 0, 1 \end{pmatrix}$
        """
        return Matrix([[1, self.a_n.subs({n: 0})], [0, 1]])

    def inflate(self, c_n: sp.Expr) -> PCF:
        r"""
        Inflates the PCF by a polynomial c.
        See LinearRecurrence.inflate for more details.
        """
        return PCF(self.recurrence.inflate(c_n))

    def deflate(self, c_n: sp.Expr) -> PCF:
        r"""
        Deflates the PCF by a polynomial c.
        See LinearRecurrence.deflate for more details.
        """
        return PCF(self.recurrence.deflate(c_n))

    def deflate_all(self) -> PCF:
        """
        Deflates the PCF to the fullest extent, by calculating the biggest $c_n$ possibly deflated
        """
        return self.deflate(content(self.a_n, self.b_n, [n]))

    def simplify(self) -> PCF:
        """Simplifies the PCF (i.e, simplifies (a, b))"""
        return PCF(self.a_n.factor().simplify(), self.b_n.factor().simplify())

    def subs(self, *args, **kwargs) -> PCF:
        """Substitutes parameters in the PCF"""
        return PCF(self.a_n.subs(*args, **kwargs), self.b_n.subs(*args, **kwargs))

    def singular_points(self) -> list[dict]:
        return [
            solution
            for solution in self.M().singular_points()
            if solution[n].is_integer and solution[n] > 0
        ]

    @batched("iterations")
    def limit(self, iterations: Batchable[int], start: int = 1) -> list[Limit]:
        r"""
        Returns the limit corresponding to calculating the PCF up to a certain depth, including $a_0$

        This is essentially $\A \cdot \prod_{i=0}^{n-1}M(s + i)$ where `n=iterations` and `s=start`.

        Args:
            iterations: The amount of multiplications to perform. Can be an integer value or a list of values.
            start: The n value of the first matrix to be multiplied (1 by default)
        Returns:
            The pcf convergence limit as defined above.
            If iterations is a list, returns a list of limits.
        """
        return self.recurrence.limit(iterations, start=start, initial_values=self.A())

    def delta(self, depth: int, L=None) -> mp.mpf:
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{q_n}^{1+\delta}$

        If the limit is not specified (`L` is None),
        then `L` is approximated as `self.limit(2 * depth).as_float()`.

        Args:
            depth: $n$
            limit: $L$
        Returns:
            the delta value as defined above.
        Raises:
            ValueError: if depth <= 0
        """

        if depth <= 0:
            raise ValueError("Cannot calculate delta up to a non-positive depth")

        if L is None:
            limit_and_L = self.limit([depth, 2 * depth])
            limit, L = limit_and_L[0], limit_and_L[1].as_float()
        else:
            limit = self.limit(depth)
        return limit.delta(L)

    def delta_sequence(self, depth: int, L: float = None) -> list[mp.mpf]:
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{q_n}^{1+\delta}$

        If the limit is not specified (`L` is None),
        then `L` is approximated as `self.limit(2 * depth).as_float()`.

        Args:
            depth: $n$
            limit: $L$
        Returns:
            the delta values for all depths up to `depth` as defined above.
        Raises:
            ValueError: if depth <= 0
        """

        if depth <= 0:
            raise ValueError("Cannot calculate delta up to a non-positive depth")
        iterations = list(range(1, depth))
        if L is None:
            limits_and_L = self.limit(iterations + [depth * 2])
            limits, L = limits_and_L[:-1], limits_and_L[-1].as_float()
        else:
            limits = self.limit(iterations)
        deltas = [limit.delta(L) for limit in limits]
        return deltas

    def kamidelta(self, depth: int = 20) -> mp.mpf:
        r"""
        Uses the Kamidelta alogrithm to predict the delta value of the PCF.
        Effectively calls kamidelta on `M`, the recurrence matrix.

        For more details, see `Matrix.kamidelta`
        """
        return self.M().kamidelta(depth)[0]
