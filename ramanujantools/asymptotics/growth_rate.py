from __future__ import annotations

import sympy as sp
from sympy.abc import t, n


class GrowthRate:
    r"""
    Represents the formal asymptotic growth rate of a solution to a linear difference equation.

    The asymptotic behavior is captured by the Birkhoff-Trjitzinsky formal series representation,
    which defines a canonical basis solution $E(n)$ as:
    $$E(n) = (n!)^{d} \cdot \lambda^n \cdot e^{Q(n)} \cdot n^{D} \cdot (\log n)^{m}$$

    This class mathematically isolates the distinct orders of infinity (the exponents and bases)
    into a structured object. It acts as an element in a Tropical Semiring, where addition (`+`)
    filters for the strictly dominant growth rate, and multiplication (`*`) algebraically
    combines the formal exponents.

    By default, an uninitialized `GrowthRate` represents the additive identity (a "zero" or
    "dead" growth), as $\lambda = 0$ collapses the entire expression to zero.

    Args:
        factorial_power: The exponent $d$ applied to the factorial term $(n!)^d$.
            Strictly dominates all other growth components.
        exp_base: The base $\lambda$ of the primary exponential growth $\lambda^n$.
            Evaluated by its absolute magnitude. A value of $0$ nullifies the entire solution.
        sub_exp: The fractional exponent $Q(n)$ applied to $e^{Q(n)}$.
            Must be strictly sub-linear (e.g., contains fractional powers like $n^{1/p}$).
        polynomial_degree: The exponent $D$ applied to the polynomial term $n^D$.
            Can be a fractional rational shift introduced by gauge transformations.
        log_power: The exponent $m$ applied to the logarithmic term $(\log n)^m$.
            Typically represents the Jordan block depth of degenerate eigenvalues.
    """

    def __init__(
        self,
        factorial_power: sp.Integer = sp.S.Zero,
        exp_base: sp.Expr = sp.S.Zero,
        sub_exp: sp.Expr = sp.S.Zero,
        polynomial_degree: sp.Expr = sp.S.Zero,
        log_power: sp.Integer = sp.S.Zero,
    ):
        self.factorial_power: sp.Expr = factorial_power
        self.exp_base: sp.Expr = exp_base
        self.sub_exp: sp.Expr = sub_exp
        self.polynomial_degree: sp.Expr = polynomial_degree
        self.log_power: int = log_power

    def __add__(self, other: GrowthRate) -> GrowthRate:
        """Addition acts as a max() filter, keeping only the dominant GrowthRate."""
        if not isinstance(other, GrowthRate):
            raise NotImplementedError("Can only add GrowthRate to GrowthRate")
        return self if self > other else other

    def __radd__(self, other: GrowthRate) -> GrowthRate:
        return self.__add__(other)

    def __mul__(self, other: GrowthRate) -> GrowthRate:
        """Strictly combines two GrowthRates by adding their formal exponents."""
        if not isinstance(other, GrowthRate):
            raise NotImplementedError("Can only multiply GrowthRate by GrowthRate")

        return GrowthRate(
            factorial_power=sp.simplify(self.factorial_power + other.factorial_power),
            exp_base=sp.simplify(self.exp_base * other.exp_base),
            sub_exp=sp.simplify(self.sub_exp + other.sub_exp),
            polynomial_degree=sp.simplify(
                self.polynomial_degree + other.polynomial_degree
            ),
            log_power=self.log_power + other.log_power,
        )

    def __rmul__(self, other: GrowthRate) -> GrowthRate:
        return self.__mul__(other)

    def __eq__(self, other: GrowthRate) -> bool:
        """Safely checks equality by proving the difference is mathematically zero."""
        if not isinstance(other, GrowthRate):
            return False

        return (
            self.factorial_power == other.factorial_power
            and self.exp_base == other.exp_base
            and self.sub_exp == other.sub_exp
            and self.polynomial_degree == other.polynomial_degree
            and self.log_power == other.log_power
        )

    def __gt__(self, other: GrowthRate) -> bool:
        if not isinstance(other, GrowthRate):
            return True

        syms = (
            getattr(self.factorial_power, "free_symbols", set())
            | getattr(self.sub_exp, "free_symbols", set())
            | getattr(other.sub_exp, "free_symbols", set())
            | getattr(self.polynomial_degree, "free_symbols", set())
            | getattr(other.polynomial_degree, "free_symbols", set())
            | getattr(other.factorial_power, "free_symbols", set())
        )
        n_sym = list(syms)[0] if syms else sp.Symbol("n")

        n_real = sp.Symbol(n_sym.name, real=True, positive=True)

        def is_greater(a, b):
            diff = sp.simplify(a - b)
            if diff.is_zero:
                return None

            diff_real = diff.subs(n_sym, n_real)
            diff_re = sp.re(diff_real)

            lim = sp.limit(diff_re, n_real, sp.oo)

            if lim == sp.oo or lim.is_positive:
                return True
            if lim == -sp.oo or lim.is_negative:
                return False

            if lim.is_number:
                try:
                    val = float(lim.evalf())
                    if val > 0:
                        return True
                    if val < 0:
                        return False
                except TypeError:
                    pass

            return None

        cmp_d = is_greater(self.factorial_power, other.factorial_power)
        if cmp_d is not None:
            return cmp_d

        cmp_lam = is_greater(sp.Abs(self.exp_base), sp.Abs(other.exp_base))
        if cmp_lam is not None:
            return cmp_lam

        cmp_Q = is_greater(self.sub_exp, other.sub_exp)
        if cmp_Q is not None:
            return cmp_Q

        cmp_D = is_greater(self.polynomial_degree, other.polynomial_degree)
        if cmp_D is not None:
            return cmp_D

        return self.log_power > other.log_power

    def __ge__(self, other: GrowthRate) -> bool:
        return self > other or self == other

    def __repr__(self) -> str:
        return (
            f"GrowthRate(factorial_power={self.factorial_power}, exp_base={self.exp_base}, "
            f"sub_exp={self.sub_exp}, polynomial_degree={self.polynomial_degree}, log_power={self.log_power})"
        )

    def __str__(self) -> str:
        return str(self.as_expr(sp.Symbol("n")))

    def as_expr(self, n: sp.Symbol) -> sp.Expr:
        """Renders the formal growth as a SymPy expression."""
        expr = (
            (sp.factorial(n) ** self.factorial_power)
            * (self.exp_base**n if self.exp_base != 0 else 0)
            * sp.exp(self.sub_exp)
            * (n**self.polynomial_degree)
            * sp.log(n) ** self.log_power
        )
        return sp.simplify(expr).rewrite(sp.factorial)

    def simplify(self) -> GrowthRate:
        """Returns a new GrowthRate with all components simplified."""
        return GrowthRate(
            factorial_power=sp.simplify(self.factorial_power),
            exp_base=sp.simplify(self.exp_base),
            sub_exp=sp.simplify(self.sub_exp),
            polynomial_degree=sp.simplify(self.polynomial_degree),
            log_power=self.log_power,
        )

    @classmethod
    def from_taylor_coefficients(
        cls, coeffs: list[sp.Expr], p: int, log_power: int = 0, factorial_power: int = 0
    ) -> GrowthRate:
        """
        Calculates the exact asymptotic bounds of a formal product by extracting
        the sub-exponential and polynomial degrees via a logarithmic Maclaurin expansion.
        """
        exp_base = sp.cancel(sp.expand(coeffs[0]))
        if exp_base == sp.S.Zero:
            return cls(exp_base=sp.S.Zero)

        precision = len(coeffs)

        x = sum(
            (coeffs[k] / exp_base) * (t**k) for k in range(1, min(precision, p + 1))
        )

        # Maclaurin series of ln(1+x) up to O(t^(p+1))
        log_series = sp.expand(
            sum(((-1) ** (j + 1) / sp.Rational(j)) * (x**j) for j in range(1, p + 1))
        )
        sub_exp, poly_deg = sp.S.Zero, sp.S.Zero

        for k in range(1, p + 1):
            c_k = sp.cancel(sp.expand(log_series.coeff(t, k)))
            if c_k != sp.S.Zero:
                if k < p:
                    power = 1 - sp.Rational(k, p)
                    sub_exp += (c_k / power) * (n**power)
                else:
                    poly_deg = c_k

        return cls(
            exp_base=exp_base,
            sub_exp=sub_exp,
            polynomial_degree=poly_deg,
            log_power=log_power,
            factorial_power=factorial_power,
        )
