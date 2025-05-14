import math

import sympy
from sympy import Poly, Symbol


class GenericPolynomial:
    @staticmethod
    def of_degree(
        deg: int, var_name: str, s: Symbol, monic: bool = False
    ) -> tuple[Poly, list[Symbol]]:
        r"""
        Creates and returns a generic polynomial of a single variable with degree 'deg' with a list of
        its generic coefficients.

        Example:
            from sympy.abc import x
            f, _ = GenericPolynomial.of_degree(deg=3, var_name='f_', s=x, monic=True)
            =>
            f = f_0 * x**0 + f_1 * x**1 + f_2 * x**2 + x**3
        """
        poly_vars = list(sympy.symbols(f"{var_name}:{deg + 1}"))
        poly = Poly(sum(poly_vars[i] * s**i for i in range(deg)), s)
        if monic:
            poly += s**deg
            poly_vars = poly_vars[:-1]
        else:
            poly += poly_vars[deg] * s**deg

        return poly, poly_vars

    @staticmethod
    def _sum_to(n: int, num_var: int):
        if num_var < 1:
            raise Exception(f"number of variables {num_var} must be at least 1")
        if num_var == 1:
            yield [n]
            return
        for i in range(n + 1):
            for sub_sum in GenericPolynomial._sum_to(n=n - i, num_var=num_var - 1):
                yield [i] + sub_sum

    @staticmethod
    def of_combined_degree(deg: int, var_name: str, variables: list[Symbol]):
        r"""
        Similar to method 'of_degree', except this is a polynomial in several variables, and we bound the combined
        degree, which is the sum of the powers of the different variables. For example combined(x**2 y**3) = 5.
        The name of the coefficient of x**m * y**k will be 'varname(m,k)'

        Example:
            from sympy.abc import x, y
            f, _ = GenericPolynomial.of_combined_degree(deg=2, var_name='f_', variables=[x, y])
            =>
            f = f_(0,0) + f_(1,0) * x + f_(0,1) * y + f_(2,0) * x**2 + f_(1,1) * x * y + f_(0,2) * y**2
        """
        poly = 0
        coefficients = []
        for combined_degree in range(deg + 1):
            for powers in GenericPolynomial._sum_to(
                n=combined_degree, num_var=len(variables)
            ):
                powers_str = ",".join([str(power) for power in powers])
                c = sympy.Symbol(f"{var_name}({powers_str})")
                monom = math.prod(s**power for s, power in zip(variables, powers))
                poly += c * monom
                coefficients.append(c)

        return poly, coefficients

    @staticmethod
    def symmetric_polynomials(*expressions) -> list:
        r"""
        Returns a list of all the symmetric polynomials in the given expressions. The k'th element is the
        sum of product of k distinct elements from the 'expressions' list.
        For example, for expressions=[x,y,z] returns 1, x+y+z, xy+yz+zx, xyz.
        """
        t = sympy.Symbol("__symmetric_polynomials")
        poly = math.prod([t + v for v in expressions], start=sympy.Poly(1, t))
        return poly.all_coeffs()

    @staticmethod
    def as_symmetric(polynomial, symm_symbols: list[Symbol], symm_var_name: str):
        r"""
        Given a polynomial in (symbols, symm_symbols), tries to find a new polynomial as a function
        of the symmetric polynomial in the symm_symbols.

        Example:
            x, y, z = sympy.symbols('x y z')
            p = (x+y)*(x*y)*z + z**2*(x**2 + y**2) + (x+y) + 1
            as_symmetric(p, [x,y], 's_') = s1*s2*z + z**2*(s1**2 - 2*s2) + s1 + 1
        """
        n = len(symm_symbols)
        symm_polynomial = 0
        symm = GenericPolynomial.symmetric_polynomials(*symm_symbols)
        s = sympy.symbols(f"{symm_var_name}:{n + 1}")

        monomial, coefficient = Poly(polynomial, *symm_symbols).LT()
        while coefficient != 0:
            ex = list(monomial.exponents) + [0]
            prod = coefficient
            for i in range(0, len(symm_symbols)):
                prod *= s[i + 1] ** (ex[i] - ex[i + 1])

            symm_polynomial += prod
            prod_subs = prod.subs({s[i]: symm[i] for i in range(1, n + 1)})
            polynomial -= prod_subs
            monomial, coefficient = Poly(polynomial, *symm_symbols).LT()

        return symm_polynomial
