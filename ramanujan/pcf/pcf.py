from mpmath import mp
import sympy as sp
from sympy.abc import n

from typing import List, Collection
from multimethod import multimethod

from ramanujan import Matrix, Limit, zero, inf, delta


def is_deflatable(a_factors, b_factors, factor):
    if n in factor.free_symbols:
        return (
            a_factors.get(factor, 0) > 0
            and b_factors.get(factor, 0) > 0
            and b_factors.get(factor.subs(n, n - 1), 0) > 0
        )
    else:
        return a_factors.get(factor, 0) > 0 and b_factors.get(factor, 0) > 1


def remove_factor(a_factors, b_factors, factor):
    a_factors[factor] -= 1
    b_factors[factor] -= 1
    b_factors[factor.subs(n, n - 1)] -= 1


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
        return content, dict(factors)

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
        return Matrix([[1, self.a_n.subs(n, 0)], [0, 1]])

    def inflate(self, c_n):
        """
        Inflates the PCF by $c_n$.

        Inflation is the process of creating an almost equivalent PCF,
        such that $a_n' = a_n * c_n, b_n' = b_n * c_n * c_{n-1}$
        """
        return PCF(self.a_n * c_n, self.b_n * c_n.subs(n, n - 1) * c_n).simplify()

    def deflate(self, c_n):
        """
        Deflates the PCF by $c_n$

        Deflation is the opposite process of inflation - or inflating by $c_n^{-1}$
        """
        return self.inflate(1 / c_n)

    def deflate_all(self):
        """
        Deflates the PCF to the fullest extent, by calculating the biggest $c_n$ possibly deflated
        """
        return self.deflate(content(self.a_n, self.b_n, [n]))

    def simplify(self):
        """Simplifies the PCF (i.e, simplifies (a, b))"""
        return PCF(self.a_n.simplify(), self.b_n.simplify())

    def subs(self, *args, **kwargs):
        """Substitutes parameters in the PCF"""
        return PCF(self.a_n.subs(*args, **kwargs), self.b_n.subs(*args, **kwargs))

    @multimethod
    def walk(self, iterations: Collection[int], start: int = 1) -> List[Limit]:
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
        m_walk = self.M().walk({n: 1}, iterations, {n: start})
        return [self.A() * mat for mat in m_walk]

    @multimethod
    def walk(self, iterations: int, start: int = 1) -> Limit:  # noqa: F811
        return self.walk([iterations], start)[0]

    @staticmethod
    def precision(m: Matrix, base: int = 10) -> int:
        """
        Returns the error in 'digits' for the PCF convergence.

        If `m` is not a result of a `PCF.walk` multiplication,
        or if the pcf does not converge this function is undefined.
        By default, `base` is 10 (for digits).
        """
        diff = abs(mp.mpq(*(m * zero())) - mp.mpq(*(m * inf())))
        return int(mp.floor(-mp.log(diff, 10)))

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
        """

        if limit is None:
            m, mlim = self.walk([depth, 2 * depth])
            limit = mlim.as_float()
        else:
            m = self.walk(depth)
        p, q = m.as_rational()
        return delta(p, q, limit)
