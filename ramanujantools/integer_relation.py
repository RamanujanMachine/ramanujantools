import sympy as sp


class IntegerRelation:
    r"""
    Class represents an integer relation, i.e, a list of integer coefficients
    such that for some real numbers, the weighted sum of the numbers roughly equals zero.

    This class supports two modes:

    If `len(self.constants) == 1`, then this class represents a relation of the form
    $0 \approx \prod_{i=0}^{N-1}a_i * p_i$, where $a_i$ are the coefficients.

    If `len(self.constants) == 2`, then this class represents a relation of the form
    $0 \approx \prod_{i=0}^{N-1}a_i * p_i - L * \prod_{i=0}^{N-1}b_i * p_i$, where $a_i$ and $b_i$ are the coefficients.

    The real numbers $p_0, \dots, p_{N-1}$ and the real number $L$ are context dependent.
    """

    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __repr__(self):
        return f"IntegerRelation({self.coefficients})"

    def coefficients_expression(self, index):
        coefficients = self.coefficients[index]
        expr = 0
        for i in range(len(coefficients)):
            expr += sp.Symbol(f"p{i}") * coefficients[i]
        return expr

    def __str__(self):
        if len(self.coefficients) == 1:
            return f"0 = {self.coefficients_expression(0)}"
        else:
            return f"0 = {self.coefficients_expression(0)} - {sp.Symbol('L') * self.coefficients_expression(1)}"
