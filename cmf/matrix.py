import copy
import math
import mpmath as mp
import sympy as sp
import operator

from sympy.abc import n, x, y


class Position(list):
    def __add__(self, other):
        return Position(map(operator.add, self, other))

    def __iadd__(self, other):
        return self + other


class Matrix(sp.Matrix):
    variables = {1: [n], 2: [x, y]}

    def __call__(self, *args):
        """
        Quick substitution method.

        For 1 args, assumes variable is n.
        For 2 args, assumes variables are (x, y).
        """
        args = list(args)
        return self.subs(list(zip(Matrix.variables[len(args)], args)))

    def gcd(self):
        """Returns gcd of the matrix"""
        return math.gcd(math.gcd(self[0], self[1]), math.gcd(self[1], self[2]))

    def reduce(self):
        """Reduces gcd from the matrix"""
        gcd = self.gcd()
        for i in range(len(self)):
            self[i] //= gcd
        return self

    def limit(self, vector=sp.Matrix([[0], [1]])):
        """Returns the limit of the matrix, i.e, the ratio of M * v for some vector v"""
        p, q = self * vector
        return sp.Float(p / q)

    def walk(self, trajectory, iterations, start=[x, y]):
        """Returns the multiplication result of walking in a certain trajectory."""
        trajectory = Position(trajectory)
        position = Position(start)
        retval = Matrix.eye(2)
        for _ in range(iterations):
            retval *= self(*position)
            position += trajectory
        return simplify(retval)

    def as_pcf(self):
        """Returns the matrix's equivalent PCF with an equal limit up to a mobius transformation"""
        from cmf import PCF

        U = Matrix([[self[1, 0], -self[0, 0]], [0, 1]])
        Uinv = Matrix([[1, self[0, 0]], [0, self[1, 0]]])
        commutated = U * self * Uinv(n + 1)
        normalized = simplify(commutated / commutated[1, 0])
        return PCF.from_matrix(normalized).inflate(self[1, 0]).deflate_all()


def simplify(matrix: Matrix) -> Matrix:
    matrix.simplify()
    return matrix
