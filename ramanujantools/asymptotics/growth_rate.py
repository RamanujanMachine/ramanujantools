import sympy as sp
from dataclasses import dataclass


@dataclass(frozen=True)
class GrowthRate:
    lambda_val: sp.Expr
    Q_n: sp.Expr
    D_val: sp.Expr
    jordan_depth: int
    d: sp.Expr | int

    def __add__(self, other):
        """Addition acts as a max() filter, keeping only the dominant GrowthRate."""
        if not isinstance(other, GrowthRate):
            return self
        return self if self > other else other

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """
        Multiplication by a rational function shifts the polynomial degree.
        Multiplication by 0 kills the term entirely.
        """
        if other == 0 or other == sp.S.Zero:
            return 0

        syms = self.Q_n.free_symbols.union(getattr(other, "free_symbols", set()))
        n = list(syms)[0] if syms else sp.Symbol("n")

        try:
            num, den = sp.numer(other), sp.denom(other)
            degree_shift = sp.degree(num, n) - sp.degree(den, n)
        except Exception:
            degree_shift = 0

        return GrowthRate(
            lambda_val=self.lambda_val,
            Q_n=self.Q_n,
            D_val=sp.simplify(self.D_val + degree_shift),
            jordan_depth=self.jordan_depth,
            d=self.d,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        """Safely checks equality by proving the difference is mathematically zero."""
        if not isinstance(other, GrowthRate):
            return False

        return (
            sp.simplify(self.d - other.d).is_zero
            and sp.simplify(self.Q_n - other.Q_n).is_zero
            and sp.simplify(self.lambda_val - other.lambda_val).is_zero
            and sp.simplify(self.D_val - other.D_val).is_zero
            and self.jordan_depth == other.jordan_depth
        )

    def __gt__(self, other):
        if not isinstance(other, GrowthRate):
            return True

        syms = (
            getattr(self.Q_n, "free_symbols", set())
            | getattr(other.Q_n, "free_symbols", set())
            | getattr(self.D_val, "free_symbols", set())
            | getattr(other.D_val, "free_symbols", set())
            | getattr(self.d, "free_symbols", set())
            | getattr(other.d, "free_symbols", set())
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

        cmp_d = is_greater(self.d, other.d)
        if cmp_d is not None:
            return cmp_d

        cmp_lam = is_greater(sp.Abs(self.lambda_val), sp.Abs(other.lambda_val))
        if cmp_lam is not None:
            return cmp_lam

        cmp_Q = is_greater(self.Q_n, other.Q_n)
        if cmp_Q is not None:
            return cmp_Q

        cmp_D = is_greater(self.D_val, other.D_val)
        if cmp_D is not None:
            return cmp_D

        return self.jordan_depth > other.jordan_depth

    def __ge__(self, other):
        return self > other or self == other
