from ramanujan import Matrix


class Vector(Matrix):
    def __init__(self, vector: list):
        vector_as_lists = list(map(lambda x: [x], vector))
        Matrix.__init__(vector_as_lists)

    @staticmethod
    def zero():
        r"""Returns the zero vector $\begin{pmatrix} 0 \cr 1 \end{pmatrix}$"""
        return Vector([0, 1])

    @staticmethod
    def inf():
        r"""Returns the infinity vector $\begin{pmatrix} 1 \cr 0 \end{pmatrix}$"""
        return Vector([1, 0])
