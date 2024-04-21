import sympy as sp
from sympy.abc import x, y

from ramanujan import Matrix
from ramanujan.cmf import CMF


class FFbar(CMF):
    r"""
    Represents a Conservative Matrix Field that was generated using the f, fbar construction:
    $a(x, y) = f(x, y) - \bar{f}(x+1, y) = f(x+1, y-1) - \bar{f}(x, y-1)$,
    $b(x) = f\bar{f}(x, 0) - f\bar{f}(0, 0) = f\bar{f}(x, y) - f\bar{f}(0, y)$,

    $Mx = \begin{pmatrix} 0, b(x) \cr 1, a(x, y) \end{pmatrix}$
    $My = \begin{pmatrix} \bar{f}(x, y), b(x) \cr 1, f(x, y) \end{pmatrix}$
    """

    @staticmethod
    def linear_condition(f, fbar) -> sp.Expr:
        r"""
        Returns the linear condition value for `f` and `fbar`.

        Functions $f(x, y), \bar{f}(x, y)$ satisfy the linear condition iff:
        $f(x+1, y-1) - \bar{f}(x, y-1) + \bar{f}(x+1, y) - f(x, y) = 0$
        """
        return sp.simplify(
            f.subs([[x, x + 1], [y, y - 1]])
            - fbar.subs(y, y - 1)
            + fbar.subs(x, x + 1)
            - f
        )

    @staticmethod
    def quadratic_condition(f, fbar) -> sp.Expr:
        r"""
        Returns the quadratic condition value for `f` and `fbar`.

        Functions $f(x, y), \bar{f}(x, y)$ satisfy the quadratic condition iff:
        $f\bar{f}(x, y) - f\bar{f}(x, 0) -f\bar{f}(0, y) + f\bar{f}(0, 0) = 0$,
        where $f\bar{f}(x, y) = f(x, y) \cdot \bar{f}(x, y)$
        """
        ffbar = sp.simplify(f * fbar)
        return sp.simplify(
            ffbar - ffbar.subs(x, 0) - ffbar.subs(y, 0) + ffbar.subs([[x, 0], [y, 0]])
        )

    def __init__(self, f, fbar):
        r"""
        Constructs an `FFbar` `CMF`

        Asserts that `f` and `fbar` functions satisfy both linear and quadratic conditions.
        """
        assert FFbar.linear_condition(f, fbar) == 0, (
            "given f and fbar do not satisfy the linear condition! f="
            + str(f)
            + ", fbar="
            + str(fbar)
        )
        assert FFbar.quadratic_condition(f, fbar) == 0, (
            "given f and fbar do not satisfy the quadratic condition! f="
            + str(f)
            + ", fbar="
            + str(fbar)
        )
        self.f = f
        """The f function of the FFbar CMF"""
        self.fbar = fbar
        """The fbar function of the FFbar CMF"""

        super().__init__(
            matrices={
                x: Matrix([[0, self.b()], [1, self.a()]]),
                y: Matrix([[self.fbar, self.b()], [1, self.f]]),
            }
        )

    def __repr__(self):
        return f"FFbar({self.f}, {self.fbar})"

    def a(self) -> sp.Expr:
        r"""
        Returns the $a(x, y)$ function as constructed in the ffbar construction:
        $a(x, y) = f(x, y) - \bar{f}(x+1, y) = f(x+1, y-1) - \bar{f}(x, y-1)$
        """
        return self.f - self.fbar.subs(x, x + 1)

    def b(self) -> sp.Expr:
        r"""
        Returns the $b(x)$ function as constructed in the ffbar construction:
        $b(x) = f\bar{f}(x, 0) - f\bar{f}(0, 0) = f\bar{f}(x, y) - f\bar{f}(0, y)$,
        where $f\bar{f}(x, y) = f(x, y) \cdot \bar{f}(x, y)$
        """
        ffbar_x_0 = (self.f * self.fbar).subs(y, 0)
        return sp.simplify(ffbar_x_0 - ffbar_x_0.subs(x, 0))
