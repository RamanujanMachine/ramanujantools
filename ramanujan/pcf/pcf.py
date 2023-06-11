import sympy as sp
from sympy.abc import n

from ramanujan import Matrix


def is_deflatable(a_factors, b_factors, factor):
    if n in factor.free_symbols:
        return (
            a_factors.get(factor, 0) > 0
            and b_factors.get(factor, 0) > 0
            and b_factors.get(factor.subs(n, n - 1), 0) > 0
        )
    else:
        return a_factors.get(factor, 0) > 0 and b_factors.get(factor, 0) > 1


def remove_factor(a_factors, b_factors, factor):
    a_factors[factor] -= 1
    b_factors[factor] -= 1
    b_factors[factor.subs(n, n - 1)] -= 1


def deflate_constant(a_constant, b_constant):
    factors = sp.factorint(sp.gcd(a_constant**2, b_constant))
    constant = 1
    for root, mul in factors.items():
        constant *= root ** (mul // 2)
    return constant


def content(a, b, variables):
    if len(a.free_symbols | b.free_symbols) == 0:
        return deflate_constant(a, b)

    def factor_list(poly, variables):
        content, factors = sp.factor_list(poly, *variables)
        return content, dict(factors)

    (a_content, a_factors), (b_content, b_factors) = map(
        lambda p: factor_list(p, variables), [a, b]
    )

    c_n = content(a_content, b_content, [])
    for factor in a_factors:
        while is_deflatable(a_factors, b_factors, factor):
            remove_factor(a_factors, b_factors, factor)
            c_n *= factor

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
        self.a_n = a_n
        self.b_n = b_n

    def __eq__(self, other):
        return (
            sp.simplify(self.a_n - other.a_n) == 0
            and sp.simplify(self.b_n - other.b_n) == 0
        )

    def __repr__(self):
        return "PCF({}, {})".format(self.a_n, self.b_n)

    def degree(self):
        return tuple(map(lambda p: sp.Poly(p).degree(), [self.a_n, self.b_n]))

    def M(self):
        """Returns the matrix that represents the PCF"""
        return Matrix([[0, self.b_n], [1, self.a_n]])

    def A(self):
        """Returns the matrix A used to calculate the limit (represents a0)"""
        return Matrix([[1, self.a_n.subs(n, 0)], [0, 1]])

    def inflate(self, c_n):
        """Inflates the PCF by c_n"""
        return PCF(self.a_n * c_n, self.b_n * c_n.subs(n, n - 1) * c_n).simplify()

    def deflate(self, c_n):
        """Deflates the PCF by c_n"""
        return self.inflate(1 / c_n)

    def deflate_all(self):
        """Deflates the PCF to the fullest extent"""
        return self.deflate(content(self.a_n, self.b_n, [n]))

    def simplify(self):
        """Simplifies the PCF (i.e, simplifies (a, b))"""
        return PCF(self.a_n.simplify(), self.b_n.simplify())

    def subs(self, substitutions):
        """Substitutes parameters in the PCF"""
        return PCF(self.a_n.subs(substitutions), self.b_n.subs(substitutions))

    def walk(self, iterations, start=1) -> Matrix:
        """Returns the matrix walk multiplication"""
        return self.M().walk({n: 1}, iterations, {n: start})

    def limit(self, depth, start=1, vector=Matrix([[0], [1]])) -> sp.Float:
        """Calculates the convergence limit of the PCF"""
        return (self.A() * self.walk(depth, start)).limit(vector)
