from __future__ import annotations

import sympy.polys.polymatrix as polymatrix

from ramanujantools import Matrix


class PolyMatrix(polymatrix.PolyMatrix):
    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], Matrix):
            return PolyMatrix.from_Matrix(*args)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __mul__(self, other) -> PolyMatrix:
        if isinstance(other, Matrix):
            if len(other.free_symbols):
                return super().__mul__(PolyMatrix(other))
            else:
                return super().__mul__(PolyMatrix(other, self.gens))

        return super().__mul__(other)

    def __rmul__(self, other) -> PolyMatrix:
        if isinstance(other, Matrix):
            return super().__rmul__(PolyMatrix(other))
        return super().__rmul__(other)

    def to_Matrix(self: PolyMatrix) -> Matrix:
        return Matrix(super().to_Matrix())
