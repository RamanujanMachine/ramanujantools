import copy
import math
import mpmath as mp
import sympy as sp
import operator

from sympy.abc import x, y


class Position(list):
    def __add__(self, other):
        return Position(map(operator.add, self, other))

    def __iadd__(self, other):
        return self + other


class Matrix(sp.Matrix):
    def __call__(self, x_value, y_value):
        return self.subs([[x, x_value], [y, y_value]])

    def gcd(self):
        return math.gcd(*self)

    def reduce(self):
        gcd = self.gcd()
        for i in range(len(self)):
            self[i] //= gcd
        return self

    def limit(self, vector=sp.Matrix([[0], [1]])):
        p, q = self * vector
        return sp.Float(p / q)

    def walk(self, trajectory, iterations, start=[x, y]):
        trajectory = Position(trajectory)
        position = Position(start)
        retval = Matrix.eye(2)
        for _ in range(iterations):
            retval *= self(*position)
            position += trajectory
        return simplify(retval)


def simplify(matrix: Matrix) -> Matrix:
    matrix.simplify()
    return matrix
