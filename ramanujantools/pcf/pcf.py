import mpmath as mp
import sympy as sp
from sympy.abc import n

from typing import Collection
from multimethod import multimethod

from ramanujantools import Matrix, Limit


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

    def __init__(self, a_n, b_n):
        """
        Initializes a PCF with `a_n` and `b_n` polynomials, for example:
        `pcf = PCF(5 + 10 * n, 1 - 9 * n**2)`
        """
        self.a_n = sp.simplify(a_n)
        """The a_n polynomial"""

        self.b_n = sp.simplify(b_n)
        """The b_n polynomial"""

    def __eq__(self, other):
        return (
            sp.simplify(self.a_n - other.a_n) == 0
            and sp.simplify(self.b_n - other.b_n) == 0
        )

    def __repr__(self):
        return f"PCF({self.a_n}, {self.b_n})"

    def degree(self):
        """
        Returns the degrees of a_n and b_n as a tuple: $(deg(a_n), deg(b_n))$
        """
        return tuple(map(lambda p: sp.Poly(p, n).degree(), [self.a_n, self.b_n]))

    def M(self):
        r"""
        Returns the matrix that represents the PCF recurrence:

        $M = \begin{pmatrix} 0, b_n \cr 1, a_n \end{pmatrix}$
        """
        return Matrix([[0, self.b_n], [1, self.a_n]])

    def A(self):
        r"""
        Returns the matrix that represents the $a_0$ part:

        $A = \begin{pmatrix} 1, a_0 \cr 0, 1 \end{pmatrix}$
        """
        return Matrix([[1, self.a_n.subs({n: 0})], [0, 1]])

    def inflate(self, c_n):
        """
        Inflates the PCF by $c_n$.

        Inflation is the process of creating an almost equivalent PCF,
        such that $a_n' = a_n * c_n, b_n' = b_n * c_n * c_{n-1}$
        """
        c_n = sp.simplify(c_n)
        return PCF(self.a_n * c_n, self.b_n * c_n.subs({n: n - 1}) * c_n).simplify()

    def deflate(self, c_n):
        """
        Deflates the PCF by $c_n$

        Deflation is the opposite process of inflation - or inflating by $c_n^{-1}$
        """
        c_n = sp.simplify(c_n)
        return self.inflate(1 / c_n)

    def deflate_all(self):
        """
        Deflates the PCF to the fullest extent, by calculating the biggest $c_n$ possibly deflated
        """
        return self.deflate(content(self.a_n, self.b_n, [n]))

    def simplify(self):
        """Simplifies the PCF (i.e, simplifies (a, b))"""
        return PCF(self.a_n.cancel().simplify(), self.b_n.cancel().simplify())

    def subs(self, *args, **kwargs):
        """Substitutes parameters in the PCF"""
        return PCF(self.a_n.subs(*args, **kwargs), self.b_n.subs(*args, **kwargs))

    def singular_points(self) -> list[dict]:
        return [
            solution
            for solution in self.M().singular_points()
            if solution[n].is_integer and solution[n] > 0
        ]

    @multimethod
    def walk(self, iterations: Collection[int], start: int = 0) -> list[Matrix]:
        r"""
        Returns the matrix corresponding to calculating the PCF up to a certain depth, including $a_0$

        This is essentially $A \cdot \prod_{i=1}^{n-1}M(i)$ where `n=iterations` if `start==0`,
        $\prod_{i=s}^{s+n-1}M(i)$ where `n=iterations` and `s = start` otherwise.

        Args:
            iterations: The amount of multiplications to perform. Can be an integer value or a list of values.
            start: The n value of the first matrix to be multiplied (1 by default)
        Returns:
            The pcf convergence limit as defined above.
            If iterations is a list, returns a list of limits.
        """
        if not all(depth >= 0 for depth in iterations):
            raise ValueError(
                f"iterations must contain only non-negative values, got {iterations}"
            )
        iterations = sorted(list(set(iterations)))
        walk_results = []
        if start == 0:
            if iterations[0] == 0:
                iterations = iterations[1:]
                walk_results.append(Matrix.eye(2))
            actual_iterations = sorted([depth - 1 for depth in iterations])
            current_results = self.M().walk({n: 1}, actual_iterations, {n: 1})
            walk_results += [self.A() * result for result in current_results]
        else:
            walk_results += self.M().walk({n: 1}, iterations, {n: start})
        return walk_results

    @multimethod
    def walk(self, iterations: int, start: int = 0) -> Matrix:  # noqa: F811
        return self.walk([iterations], start)[0]

    @multimethod
    def limit(self, iterations: Collection[int], start: int = 0) -> list[Limit]:
        r"""
        Returns the limit corresponding to calculating the PCF up to a certain depth, including $a_0$

        This is essentially $A \cdot \prod_{i=0}^{n-1}M(s + i)$ where `n=iterations` and `s=start`

        Args:
            iterations: The amount of multiplications to perform. Can be an integer value or a list of values.
            start: The n value of the first matrix to be multiplied (1 by default)
        Returns:
            The pcf convergence limit as defined above.
            If iterations is a list, returns a list of limits.
        """

        def walk_function(iterations):
            return self.walk(iterations, start)

        return Limit.walk_to_limit(iterations, walk_function)

    @multimethod
    def limit(self, iterations: int, start: int = 0) -> Limit:  # noqa: F811
        return self.limit([iterations], start)[0]

    def delta(self, depth, limit=None):
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{q_n}^{1+\delta}$

        If limit is not specified (i.e, limit is None),
        then limit is approximated as limit = self.limit(2 * depth)

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

        if limit is None:
            m, mlim = self.limit([depth, 2 * depth])
            limit = mlim.as_float()
        else:
            m = self.limit(depth)
        return m.delta(limit)

    def delta_sequence(self, depth: int, limit: float = None):
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{q_n}^{1+\delta}$

        If limit is not specified (i.e, limit is None),
        then limit is approximated as the limit at 2*depth.

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

        if limit is None:
            limit = self.limit(2 * depth).as_float()

        deltas = []
        prev = Matrix.eye(2)
        m = self.A()
        deltas.append(Limit(m, prev).delta(limit))

        for i in range(1, depth):
            prev = m
            m *= self.M()({n: i})
            deltas.append(Limit(m, prev).delta(limit))

        return deltas

    def kamidelta(self, depth: int = 20) -> mp.mpf:
        r"""
        Uses the Kamidelta alogrithm to predict the delta value of the PCF.
        Effectively calls kamidelta on `M`, the recurrence matrix.

        For more details, see `Matrix.kamidelta`
        """
        return self.M().kamidelta(depth)[0]
