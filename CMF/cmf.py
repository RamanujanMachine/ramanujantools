import sympy as sp
from sympy.abc import x, y

from matrix import Matrix, simplify


class CMF:
    def __init__(self, Mx: Matrix, My: Matrix):
        self.m_Mx = Mx
        self.m_My = My
        Mxy = simplify(self.Mx(x, y) * self.My(x + 1, y))
        Myx = simplify(self.My(x, y) * self.Mx(x, y + 1))
        if simplify(Mxy - Myx) != Matrix([[0, 0], [0, 0]]):
            raise ValueError("The given Mx and My matrices are not conserving!")

    def Mx(self, x, y) -> Matrix:
        return self.m_Mx(x, y)

    def My(self, x, y) -> Matrix:
        return self.m_My(x, y)

    def trajectory_matrix(self, trajectory) -> Matrix:
        m = self.Mx(x, y).walk([1, 0], trajectory[0])
        m *= self.My(x + trajectory[0], y).walk([0, 1], trajectory[1])
        return simplify(m)

    def walk(self, trajectory, iterations, start=[1, 1]) -> Matrix:
        m = self.trajectory_matrix(trajectory)
        return m.walk(trajectory, iterations // sum(trajectory), start)

    def limit(self, trajectory, iterations, start=[1,1], vector=Matrix([[0], [1]])) -> sp.Float:
        return self.walk(trajectory, iterations, start).limit(vector)
