import sympy as sp
import mpmath as mp

from ramanujan import Matrix


class Vector(Matrix):
    def __init__(self, vector: list):
        vector_as_lists = list(map(lambda x: [x], vector))
        Matrix.__init__(vector_as_lists)

    def ratio(self):
        assert len(self) == 2, "Ratio only supported for vectors of length 2"
        return sp.Float(self[0] / self[1], mp.mp.dps)

    @staticmethod
    def zero():
        r"""Returns the zero vector $\begin{pmatrix} 0 \cr 1 \end{pmatrix}$"""
        return Vector([0, 1])

    @staticmethod
    def inf():
        r"""Returns the infinity vector $\begin{pmatrix} 1 \cr 0 \end{pmatrix}$"""
        return Vector([1, 0])
