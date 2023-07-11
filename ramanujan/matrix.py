import math
import sympy as sp


class Matrix(sp.Matrix):
    def __call__(self, substitutions):
        """Same as 'subs', but in a math-like syntax."""
        return self.subs(substitutions)

    def gcd(self):
        """Returns gcd of the matrix"""
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
        return retval.simplify()

    def as_pcf(self, deflate_all=True):
        """Returns the matrix's equivalent PCF with an equal limit up to a mobius transformation"""
        from ramanujan.pcf import PCFFromMatrix

        return PCFFromMatrix.convert(self, deflate_all)
