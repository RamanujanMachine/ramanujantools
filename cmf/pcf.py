import math
import sympy as sp
from sympy.abc import n

from cmf import Matrix


def is_pcf(M: Matrix):
    return M[0, 0] == 0 and M[1, 0] == 1


def remove_root(factorization, root):
    factorization[root] -= 1
    if factorization[root] == 0:
        del factorization[root]


def deflate_coeffs(a_coeff, b_coeff):
    possible_roots = sp.factorint(math.gcd(a_coeff**2, b_coeff))
    deflateable = 1
    for root, mul in possible_roots.items():
        deflateable *= root ** (mul // 2)
    return a_coeff // deflateable, b_coeff // (deflateable**2)


def deflate_attempt(a_roots, b_roots):
    for root in a_roots:
        if root in b_roots and root + 1 in b_roots:
            remove_root(a_roots, root)
            remove_root(b_roots, root)
            remove_root(b_roots, root + 1)
            return True
    return False


def construct_from_roots(factorization, leading_coefficient):
    retval = leading_coefficient
    print(factorization)
    for root, mul in factorization.items():
        retval *= (n - root) ** mul
    return retval


class PCF:
    @classmethod
    def from_matrix(cls, M: Matrix):
        if not is_pcf(M):
            raise ValueError("The given matrix M is not of a PCF form!")
        return cls(M[1, 1], M[0, 1])

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
        return PCF(self.m_a * c_n, self.m_b * c_n.subs(n, n - 1) * c_n)

    def deflate_all(self):
        a_roots = sp.roots(self.m_a)
        b_roots = sp.roots(self.m_b)
        while deflate_attempt(a_roots, b_roots):
            pass
        a_coeff, b_coeff = deflate_coeffs(sp.LC(self.m_a), sp.LC(self.m_b))
        return PCF(
            construct_from_roots(a_roots, a_coeff),
            construct_from_roots(b_roots, b_coeff),
        )

    def deflate(self, c_n=None):
        if c_n:
            return self.inflate(1 / c_n)
        return self.deflate_all()

    def simplify(self):
        return PCF(self.m_a.simplify(), self.m_b.simplify())

    def subs(self, substitutions):
        return PCF.from_matrix(self.m_M.subs(substitutions))

    def walk(self, iterations, start=1) -> Matrix:
        return self.M().walk([1], iterations, start)

    def limit(self, depth, start=[1], vector=Matrix([[0], [1]])) -> sp.Float:
        return (self.A() * self.walk(depth, start)).limit(vector)
