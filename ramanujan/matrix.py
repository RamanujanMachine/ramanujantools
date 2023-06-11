import math
import sympy as sp


class Matrix(sp.Matrix):
    def __call__(self, substitutions):
        """Same as 'subs', but in a math-like syntax."""
        return self.subs(substitutions)

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

    def walk(self, trajectory, iterations, start):
        """Returns the multiplication result of walking in a certain trajectory."""
        position = start
        retval = Matrix.eye(2)
        for _ in range(iterations):
            retval *= self.subs(position)
            position = {key: trajectory[key] + value for key, value in position.items()}
        return simplify(retval)

    def as_pcf(self, deflate_all=True):
        """Returns the matrix's equivalent PCF with an equal limit up to a mobius transformation"""
        from ramanujan.pcf import PCF
        from sympy.abc import n

        U = Matrix([[self[1, 0], -self[0, 0]], [0, 1]])
        Uinv = Matrix([[1, self[0, 0]], [0, self[1, 0]]])
        commutated = U * self * Uinv({n: n + 1})
        normalized = simplify(commutated / commutated[1, 0])
        pcf = PCF.from_matrix(normalized).inflate(self[1, 0])
        if deflate_all:
            pcf = pcf.deflate_all()
        return pcf


def simplify(matrix: Matrix) -> Matrix:
    matrix.simplify()
    return matrix
