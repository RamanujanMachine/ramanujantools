import math

import mpmath as mp
import sympy as sp


class Matrix(sp.Matrix):
    """
    Represens a marix.

    Inherits from sympy's matrix and supports all of its methods and operators:
    https://docs.sympy.org/latest/modules/matrices/matrices.html
    """

    def __call__(self, *args, **kwargs):
        """
        Substitutes variables in the matrix, in a more math-like syntax.

        Calls the underlying sympy `subs` method:
        https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs
        """
        return self.subs(*args, **kwargs)

    def gcd(self):
        """
        Returns the gcd of the matrix
        """
        import functools

        return functools.reduce(math.gcd, self)

    def reduce(self):
        """Reduces gcd from the matrix"""
        gcd = self.gcd()
        for i in range(len(self)):
            self[i] //= gcd
        return self

    def simplify(self):
        """Returns a simplified version of matrix"""
        return Matrix(sp.simplify(self))

    def walk(self, trajectory, iterations, start):
        r"""
        Returns the multiplication result of walking in a certain trajectory.

        The `walk` operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=self`, `(t_0, ..., t_k)=trajectory`, `n=iterations` and `(s_0, ..., s_k)=start`.

        This is a generalization of the basic (and most common) case $\prod_{i=0}^{n-1}M(s+i)$,
        where `M=self`, `n=iterations` and `s=start`.

        Args:
            trajectory: the trajectory of a single step in the walk, as defined above.
            iterations: the amount of multiplications to perform
            start: the starting point of the matrix multiplication
        Returns:
            the walk multiplication as defined above.
        """
        position = start
        retval = Matrix.eye(2)
        for _ in range(iterations):
            retval *= self.subs(position)
            position = {key: trajectory[key] + value for key, value in position.items()}
        return retval.simplify()

    def ratio(self):
        assert len(self) == 2, "Ratio only supported for vectors of length 2"
        return sp.Float(self[0] / self[1], mp.mp.dps)

    def as_pcf(self, deflate_all=True):
        """
        Converts a `Matrix` to an equivalent `PCF`

        Args:
            deflate_all: if `True`, the function will also deflate the returned PCF to the fullest.
        Returns:
            a `PCFFRomMatrix` object, containing a `PCF` whose limit is equal to
            a mobius transform of the original `Matrix`.
        """
        from ramanujan.pcf import PCFFromMatrix

        return PCFFromMatrix(self, deflate_all)


def zero():
    r"""Returns the zero vector $\begin{pmatrix} 0 \cr 1 \end{pmatrix}$"""
    return Matrix([0, 1])


def inf():
    r"""Returns the infinity vector $\begin{pmatrix} 1 \cr 0 \end{pmatrix}$"""
    return Matrix([1, 0])
