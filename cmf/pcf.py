import sympy as sp
from sympy.abc import n

from cmf import Matrix


def is_pcf(M: Matrix):
    return M[0, 0] == 1 and M[1, 0] == 0


class PCF:
    @classmethod
    def from_matrix(cls, M: Matrix):
        if not is_pcf(M):
            raise ValueError("The given matrix M is not of a PCF form!")
        return cls(matrix[1, 1], matrix[0, 1])

    def __init__(self, a_n, b_n):
        self.m_a = a_n
        self.m_b = b_n

    def __eq__(self, other):
        return self.m_a == other.m_a and self.m_b == other.m_b

    def __repr__(self):
        return "PCF({}, {})".format(self.m_a, self.m_b)

    def M(self):
        return Matrix([[0, self.m_b], [1, self.m_a]])

    def A(self):
        return Matrix([[1, self.m_a.subs(n, 0)], [0, 1]])

    def inflate(self, c_n):
        return PCF(self.m_a * c_n, self.m_b * c_n.subs(n, n-1) * c_n)

    def subs(self, substitutions):
        return PCF.from_matrix(self.m_M.subs(substitutions))

    def walk(self, iterations, start=1) -> Matrix:
        return self.M().walk([1], iterations, start)

    def limit(self, depth, start=[1], vector=Matrix([[0], [1]])) -> sp.Float:
        return (self.A() * self.walk(depth, start)).limit(vector)
