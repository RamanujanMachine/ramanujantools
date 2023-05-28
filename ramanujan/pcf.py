import math
import sympy as sp
from sympy.abc import n

from ramanujan import Matrix


def remove_root(factorization, root):
    """Given a factorization for n, returns the factorization of n / root"""
    factorization[root] -= 1
    if factorization[root] == 0:
        del factorization[root]


def deflate_coeffs(a_coeff, b_coeff):
    """Deflates the coefficients of (a, b)"""
    possible_roots = sp.factorint(math.gcd(a_coeff**2, b_coeff))
    deflateable = 1
    for root, mul in possible_roots.items():
        deflateable *= root ** (mul // 2)
    return a_coeff // deflateable, b_coeff // (deflateable**2)


def deflate_single_root(a_roots, b_roots):
    """Attempts to find a root that can be deflated for (a, b)"""
    for root in a_roots:
        if root in b_roots and root + 1 in b_roots:
            remove_root(a_roots, root)
            remove_root(b_roots, root)
            remove_root(b_roots, root + 1)
            return True
    return False


def construct_poly(roots, coeff):
    """Constructs a polynomial from it's roots and a leading coefficient"""
    retval = coeff
    for root, mul in roots.items():
        retval *= (n - root) ** mul
    return retval.simplify()


class PCF:
    @staticmethod
    def is_pcf(M: Matrix):
        """Checks if the given matrix is of a PCF form"""
        return M[0, 0] == 0 and M[1, 0] == 1

    @classmethod
    def from_matrix(cls, M: Matrix):
        """Constructs a PCF from a matrix"""
        if not PCF.is_pcf(M):
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
        """Returns the matrix that represents the PCF"""
        return Matrix([[0, self.m_b], [1, self.m_a]])

    def A(self):
        """Returns the matrix A used to calculate the limit (represents a0)"""
        return Matrix([[1, self.m_a.subs(n, 0)], [0, 1]])

    def inflate(self, c_n):
        """Inflates the PCF by c_n"""
        return PCF(self.m_a * c_n, self.m_b * c_n.subs(n, n - 1) * c_n).simplify()

    def deflate(self, c_n):
        """Deflates the PCF by c_n"""
        return self.inflate(1 / c_n)

    def deflate_all(self):
        """Deflates the PCF to the fullest extent"""
        a_coeff, b_coeff = deflate_coeffs(sp.LC(self.m_a), sp.LC(self.m_b))
        a_roots = sp.roots(self.m_a)
        b_roots = sp.roots(self.m_b)
        while deflate_single_root(a_roots, b_roots):
            pass
        return PCF(
            construct_poly(a_roots, a_coeff),
            construct_poly(b_roots, b_coeff),
        )

    def simplify(self):
        """Simplifies the PCF (i.e, simplifies (a, b))"""
        return PCF(self.m_a.simplify(), self.m_b.simplify())

    def subs(self, substitutions):
        """Substitutes variables in the PCF"""
        return PCF(self.m_a.subs(substitutions), self.m_b.subs(substitutions))

    def walk(self, iterations, start=1) -> Matrix:
        """Returns the matrix walk multiplication"""
        return self.M().walk({n: 1}, iterations, {n: start})

    def limit(self, depth, start=1, vector=Matrix([[0], [1]])) -> sp.Float:
        """Calculates the convergence limit of the PCF"""
        return (self.A() * self.walk(depth, start)).limit(vector)
