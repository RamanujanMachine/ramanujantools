from sympy.abc import n

from ramanujan import Matrix
from ramanujan.pcf import PCF


class PCFFromMatrix:
    def __init__(self, matrix: Matrix, deflate_all=True):
        """
        Constructs a PCF from a matrix using a a coboundry matrix U.

        The created PCF has the same convergence limit of the original matrix up to a certain mobius transformation.
        """
        assert (
            2 == matrix.rows and 2 == matrix.cols
        ), "Conversion of arbitrary matrix to PCF is only supported for 2x2 matrices!"

        U = Matrix([[matrix[1, 0], -matrix[0, 0]], [0, 1]])
        Uinv = Matrix([[1, matrix[0, 0]], [0, matrix[1, 0]]])
        commutated = U * matrix * Uinv({n: n + 1})
        normalized = (commutated / commutated[1, 0]).simplify()
        if not (normalized[0, 0] == 0 and normalized[1, 0] == 1):
            raise ValueError(
                f"An error has occured when converting matrix {matrix} into a pcf"
            )
        pcf = PCF(normalized[1, 1], normalized[0, 1])
        pcf = pcf.inflate(matrix[1, 0])
        if deflate_all:
            pcf = pcf.deflate_all()

        self.pcf = pcf
        """The pcf itself equivalent to the original matrix up to a mobius transform"""

        self.U = U
        """The coboundry matrix used to generate the pcf"""

    def __eq__(self, other):
        return self.pcf == other.pcf and self.U == other.U

    def relative_limit(self):
        r"""
        Returns the mobius transform between the original matrix limit and the pcf relative_limit.

        Let R = relative_limit(self), P = walk(pcf) and M the original matrix. then:
        $P = R \cdot M$
        """
        return self.pcf.A() * self.U.subs(n, 1)
