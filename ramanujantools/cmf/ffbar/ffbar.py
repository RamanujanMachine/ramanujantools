from __future__ import annotations

import sympy as sp
from sympy.abc import x, y

from ramanujantools import Matrix, Position
from ramanujantools.cmf import CMF


class FFbar(CMF):
    r"""
    Represents a 2D Conservative Matrix Field that was generated using the f, fbar construction:
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
        Constructs an FFbar CMF.

        Args:
            f: the $f$ function
            fbar: the $\bar{f}$ function

        Raises:
            ValueError: if f and fbar do not satisfy the linear condition or the quadratic condition.
        """
        if FFbar.linear_condition(f, fbar) != 0:
            raise ValueError(
                f"given f and fbar do not satisfy the linear condition! f={f}, fbar={fbar}"
            )
        if FFbar.quadratic_condition(f, fbar) != 0:
            raise ValueError(
                f"given f and fbar do not satisfy the quadratic condition! f={f}, fbar={fbar}"
            )

        self.f = f
        """The f function of the FFbar CMF"""
        self.fbar = fbar
        """The fbar function of the FFbar CMF"""

        super().__init__(
            matrices={
                x: Matrix([[0, self.b()], [1, self.a()]]),
                y: Matrix([[self.fbar, self.b()], [1, self.f]]),
            },
            validate=False,
        )

    def __repr__(self):
        return f"FFbar({self.f}, {self.fbar})"

    def subs(self, substitutions: Position) -> FFbar:
        self._validate_axes_substitutions(substitutions)
        return FFbar(self.f.subs(substitutions), self.fbar.subs(substitutions))

    @staticmethod
    def A(f, fbar) -> sp.Expr:
        r"""
        Returns the $a(x, y)$ function as constructed in the ffbar construction:
        $a(x, y) = f(x, y) - \bar{f}(x+1, y) = f(x+1, y-1) - \bar{f}(x, y-1)$
        """
        return f - fbar.subs(x, x + 1)

    def a(self) -> sp.Expr:
        r"""
        Returns the $a(x, y)$ function for this FFbar CMF.
        """
        return FFbar.A(self.f, self.fbar)

    @staticmethod
    def B(f, fbar) -> sp.Expr:
        r"""
        Returns the $b(x)$ function as constructed in the ffbar construction:
        $b(x) = f\bar{f}(x, 0) - f\bar{f}(0, 0) = f\bar{f}(x, y) - f\bar{f}(0, y)$,
        where $f\bar{f}(x, y) = f(x, y) \cdot \bar{f}(x, y)$
        """
        ffbar_x_0 = (f * fbar).subs(y, 0)
        return sp.simplify(ffbar_x_0 - ffbar_x_0.subs(x, 0))

    def b(self) -> sp.Expr:
        r"""
        Returns the $b(x)$ function for this FFbar CMF.
        """
        return FFbar.B(self.f, self.fbar)
