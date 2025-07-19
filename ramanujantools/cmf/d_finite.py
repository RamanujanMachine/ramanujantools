from __future__ import annotations

from abc import ABC, abstractmethod

import sympy as sp

from ramanujantools import Matrix
from ramanujantools.cmf import CMF

from functools import lru_cache


theta = sp.symbols("theta")


class DFinite(CMF, ABC):
    """
    Represents a D-finite CMF, which is a base class for constructing CMFs that are derived from D-finite functions.
    In order to implement a D-finite CMF, one must define the following methods:
    1. axes: Returns the axes and their signs (directions) for the D-finite CMF.
    2. differential_equation: Returns the differential equation for the D-finite CMF.
    3. construct_matrix: Given a symbol, constructs the matrix for the D-finite CMF.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs the D-finite CMF.
        The D-finite CMF is a base class for constructing CMFs that are derived from D-finite functions.
        """
        matrices, negative_matrices = self.construct_matrices(*args, **kwargs)
        super().__init__(
            matrices=matrices,
            _negative_matrices_cache=negative_matrices,
            validate=False,
        )

    @classmethod
    @abstractmethod
    def axes_and_signs(cls, *args, **kwargs) -> dict[sp.Expr, bool]:
        """
        Function returns the axes and their signs for the D-finite CMF.
        The signs are used to determine the direction of the axes in the CMF,
        and are derived from the contiguous relations of the D-finite function.
        """
        pass

    @classmethod
    @abstractmethod
    def differential_equation(cls, *args, **kwrags) -> sp.Poly:
        """
        Returns the differential equation for the D-finite CMF.
        """
        pass

    @classmethod
    @abstractmethod
    def construct_matrix(
        cls, theta_matrix: Matrix, axis: sp.Symbol, *args, **kwargs
    ) -> Matrix:
        """
        Given a symbol, constructs the matrix for the D-finite CMF.
        Note that if axes[symbol] is False, then the matrix should be used to decrement the axis by one.
        """
        pass

    @classmethod
    @lru_cache
    def construct_matrices(
        cls, *args, **kwargs
    ) -> tuple[dict[sp.Expr, Matrix], dict[sp.Expr, Matrix]]:
        """
        Constructs the matrices for the D-finite CMF.
        """
        d_poly = cls.differential_equation(*args, **kwargs)

        if not (d_poly.free_symbols).issuperset(
            cls.axes_and_signs(*args, **kwargs).keys()
        ):
            raise ValueError(
                "The differential equation free symbols does not contain the axes!"
            )

        d_poly_monic = sp.Poly(d_poly / sp.LC(d_poly), theta)
        theta_matrix = Matrix.companion(d_poly_monic)

        matrices = {}
        negative_matrices = {}
        for axis, sign in cls.axes_and_signs(*args, **kwargs).items():
            matrix = cls.construct_matrix(theta_matrix, axis, *args, **kwargs)
            if sign:
                matrices[axis] = matrix
            else:
                negative_matrices[axis] = matrix
                matrices[axis] = matrix.subs({axis: axis + 1}).inverse().factor()
        return matrices, negative_matrices
