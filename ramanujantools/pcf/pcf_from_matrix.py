from sympy.abc import n

from ramanujantools import Matrix
from ramanujantools.pcf import PCF


class PCFFromMatrix:
    def __init__(self, matrix: Matrix, deflate_all=True):
        """
        Constructs a PCF from a matrix using a a coboundry matrix U.
        The created PCF has the same convergence limit of the original matrix up to a certain mobius transformation.

        Args:
            matrix: The matrix to convert to a PCF.
            deflate_all: If True, will deflate the PCF as much as possible.

        Raises:
            ValueError: If matrix is not a 2x2 matrix, or if the matrix is not coboundry to a pcf.
        """
        if 2 != matrix.rows or 2 != matrix.cols:
            raise ValueError(
                f"Conversion of arbitrary matrix to PCF is only supported for 2x2 matrices, "
                f"got a {matrix.rows}x{matrix.cols} matrix"
            )

        U = Matrix([[matrix[1, 0], -matrix[0, 0]], [0, 1]])
        Uinv = Matrix([[1, matrix[0, 0]], [0, matrix[1, 0]]])
        commutated = U * matrix * Uinv({n: n + 1})
        reduced = (commutated / commutated[1, 0]).simplify()
        if not (reduced[0, 0] == 0 and reduced[1, 0] == 1):
            raise ValueError("Given matrix is not coboundry to a PCF")
        pcf = PCF(reduced[1, 1], reduced[0, 1])
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
        return self.pcf.A() * self.U.subs({n: 1})
