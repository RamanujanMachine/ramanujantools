import sympy as sp
from sympy.abc import n

from ramanujan import Matrix


def is_deflatable(a_roots, b_roots, root, shift=1):
    return (
        a_roots.get(root, 0) > 0
        and b_roots.get(root, 0) > 0
        and b_roots.get(root + shift, 0) > (1 if shift == 0 else 0)
    )


def deflate_root(a_roots, b_roots, root, shift=1):
    a_roots[root] -= 1
    b_roots[root] -= 1
    b_roots[root + shift] -= 1


def deflate_constant(a_constant, b_constant):
    factors = sp.factorint(sp.gcd(a_constant**2, b_constant))
    constant = 1
    for root, mul in factors.items():
        constant *= root ** (mul // 2)
    return constant


def deflate_lead(a_leading_expression, b_leading_expression):
    (a_constant, a_factors), (b_constant, b_factors) = map(
        sp.factor_list, [a_leading_expression, b_leading_expression]
    )
    c = deflate_constant(a_constant, b_constant)
    a_factors, b_factors = map(dict, [a_factors, b_factors])
    for factor in a_factors:
        while is_deflatable(a_factors, b_factors, factor, 0):
            deflate_root(a_factors, b_factors, factor, 0)
            c *= factor
    return sp.simplify(c)


def content(a_poly, b_poly):
    a_roots, b_roots = map(lambda p: sp.roots(p, n), [a_poly, b_poly])
    c_n = deflate_lead(*map(lambda p: sp.LC(p, n), [a_poly, b_poly]))
    for root in a_roots:
        while is_deflatable(a_roots, b_roots, root):
            deflate_root(a_roots, b_roots, root)
            c_n *= n - root
    return sp.simplify(c_n)


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
        return (
            sp.simplify(self.m_a - other.m_a) == 0
            and sp.simplify(self.m_b - other.m_b) == 0
        )

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
        return self.deflate(content(self.m_a, self.m_b))

    def simplify(self):
        """Simplifies the PCF (i.e, simplifies (a, b))"""
        return PCF(self.m_a.simplify(), self.m_b.simplify())

    def subs(self, substitutions):
        """Substitutes parameters in the PCF"""
        return PCF(self.m_a.subs(substitutions), self.m_b.subs(substitutions))

    def walk(self, iterations, start=1) -> Matrix:
        """Returns the matrix walk multiplication"""
        return self.M().walk({n: 1}, iterations, {n: start})

    def limit(self, depth, start=1, vector=Matrix([[0], [1]])) -> sp.Float:
        """Calculates the convergence limit of the PCF"""
        return (self.A() * self.walk(depth, start)).limit(vector)
