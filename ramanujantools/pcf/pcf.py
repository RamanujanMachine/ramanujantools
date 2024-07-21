import sympy as sp
from sympy.abc import n

from typing import Dict, List, Collection
from multimethod import multimethod
import re

from ramanujantools import Matrix, Limit


def is_deflatable(a_factors, b_factors, factor):
    if n in factor.free_symbols:
        return (
            a_factors.get(factor, 0) > 0
            and b_factors.get(factor, 0) > 0
            and b_factors.get(factor.subs({n: n - 1}), 0) > 0
        )
    else:
        return a_factors.get(factor, 0) > 0 and b_factors.get(factor, 0) > 1


def remove_factor(a_factors, b_factors, factor):
    a_factors[factor] -= 1
    b_factors[factor] -= 1
    b_factors[factor.subs({n: n - 1})] -= 1


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

# for as_latex method
def python_to_latex(python_str):
    # Convert '**' to '^'
    latex_str = python_str.replace('**', '^')
    
    # Convert '*' to '\cdot' (for multiplication)
    latex_str = latex_str.replace('*', '') # r'\cdot '
    
    # Convert '/' to '\frac{}{}' (for division)
    # Use regular expressions to find the division operation
    def convert_division(match):
        numerator = match.group(1)
        denominator = match.group(2)
        return r'\frac{' + numerator + '}{' + denominator + '}'
    latex_str = re.sub(r'(\w+)\s*/\s*(\w+)', convert_division, latex_str)
    return latex_str

def generic_recursive_fraction(level, depth):
    if level == depth:
        return r'\ddots + \cfrac{b_n}{a_n + \ddots}'
    else:
        a_i = f'a_{level}'
        b_i = f'b_{level + 1}'
        return rf'{a_i} + \cfrac{{{b_i}}}{{{generic_recursive_fraction(level + 1, depth)}}}'

def recursive_fraction(level, depth, a_n, b_n):
    if level == depth:
        return rf'\ddots + \cfrac{{{b_n}}}{{{a_n} + \ddots}}'
    else:
        a_val = a_n.subs(n, level)
        b_val = b_n.subs(n, level + 1)
        return rf'{a_val} + \cfrac{{{b_val}}}{{{recursive_fraction(level + 1, depth, a_n, b_n)}}}'


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
    
    @multimethod
    def as_latex(self, depth: int = 3, start: int = 1) -> str:
        """
        Returns the continued fraction as a string in LaTeX format.
        Note: result should be printed to obtain actual LaTex string format.

        Args:
            depth: The index to display up to.
            start: The index from which to display.
        Returns:
            The LaTeX string for the continued fraction (python representation, i.e. '\' is '\\').
        """
        if start != 1:
            result = rf'\cfrac{{{self.b_n.subs(n, start)}}}{{{recursive_fraction(start, depth, self.a_n, self.b_n)}}}'
        else:
            result = rf'{self.a_n.subs(n, 0)} + \cfrac{{{self.b_n.subs(n, 1)}}}{{{recursive_fraction(start, depth, self.a_n, self.b_n)}}}'
        return python_to_latex(result)
    
    @multimethod
    @staticmethod
    def as_latex(depth: int = 3, start: int = 1) -> str:
        """
        Returns a generic continued fraction as a string in LaTeX format,
        with symbols $a_n$ and $b_n$ as the partial denominators and numerators, respectively.
        Args:
            depth: The index to display up to.
            start: The index from which to display.
        Returns:
            The LaTeX string for the continued fraction (python representation, i.e. '\' is '\\').
            e.g. 'a_0 + \\cfrac{b_1}{a_1 + \\cfrac{b_2}{a_2 + \\cfrac{b_3}{\\ddots + \\cfrac{b_n}{a_n + \\ddots}}}}'.
        """
        if start != 1:
            result = rf'\cfrac{{b_{start}}}{{{generic_recursive_fraction(start, depth)}}}'
        else:
            result = rf'a_0 + \cfrac{{b_1}}{{{generic_recursive_fraction(start, depth)}}}'
        return python_to_latex(result)
    
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

    def singular_points(self) -> List[Dict]:
        return [
            solution
            for solution in self.M().singular_points()
            if solution[n].is_integer and solution[n] > 0
        ]

    @multimethod
    def walk(self, iterations: Collection[int], start: int = 0) -> List[Matrix]:
        r"""
        Returns the matrix corresponding to calculating the PCF up to a certain depth, including $a_0$

        This is essentially $A \cdot \prod_{i=0}^{n-1}M(s + i)$ where `n=iterations` and `s=start`

        Args:
            iterations: The amount of multiplications to perform. Can be an integer value or a list of values.
            start: The n value of the first matrix to be multiplied (1 by default).
        Returns:
            The pcf convergence limit as defined above.
            If iterations is a list, returns a list of limits.
        """
        if not all(depth > 0 for depth in iterations):
            raise ValueError(
                f"iterations must contain only positive values, got {iterations}"
            )
        if start == 0:
            return [
                self.A() * matrix
                for matrix in self.M().walk(
                    {n: 1}, [iteration - 1 for iteration in iterations], {n: 1}
                )
            ]
        return self.M().walk({n: 1}, iterations, {n: start})

    @multimethod
    def walk(self, iterations: int, start: int = 0) -> Matrix:  # noqa: F811
        return self.walk([iterations], start)[0]

    @multimethod
    def limit(self, iterations: Collection[int], start: int = 0) -> List[Limit]:
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
        return list(map(lambda matrix: Limit(matrix), self.walk(iterations, start)))

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
        m = self.A()
        deltas.append(Limit(m).delta(limit))

        for i in range(1, depth):
            m *= self.M()({n: i})
            deltas.append(Limit(m).delta(limit))

        return deltas
