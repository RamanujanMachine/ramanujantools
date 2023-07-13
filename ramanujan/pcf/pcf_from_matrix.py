from sympy.abc import n

from ramanujan import Matrix
from ramanujan.pcf import PCF


class PCFFromMatrix:
    def __init__(self, matrix: Matrix, deflate_all=True):
        """Constructs a PCF from a matrix"""
        U = Matrix([[matrix[1, 0], -matrix[0, 0]], [0, 1]])
        Uinv = Matrix([[1, matrix[0, 0]], [0, matrix[1, 0]]])
        commutated = U * matrix * Uinv({n: n + 1})
        normalized = (commutated / commutated[1, 0]).simplify()
        if not (normalized[0, 0] == 0 and normalized[1, 0] == 1):
            raise ValueError(
                f"An error has occured when converting matrix {matrix} into a pcf"
            )
        pcf = PCF(normalized[1, 1], normalized[0, 1])
        pcf.inflate(matrix[1, 0])
        if deflate_all:
            pcf = pcf.deflate_all()
        self.pcf = pcf
        self.U = U

    def relative_limit(self):
        """Returns the mobius transform between the original matrix limit and the pcf limit"""
        return self.pcf.A() * self.U.subs(n, 1)
