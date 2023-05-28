import sympy as sp
from sympy.abc import x, y

from ramanujan import Matrix, Position, simplify


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

    def subs(self, substitutions):
        return CMF(self.m_Mx.subs(substitutions), self.m_My.subs(substitutions))

    def trajectory_matrix(self, trajectory) -> Matrix:
        m = self.m_Mx.walk({x: 1, y: 0}, trajectory[0], {x: x, y: y})
        m *= self.m_My.walk({x: 0, y: 1}, trajectory[1], {x: x + trajectory[0], y: y})
        return simplify(m)

    def walk(self, trajectory, iterations, start=[1, 1]) -> Matrix:
        m = self.trajectory_matrix(trajectory)
        return m.walk(
            Position.from_list([x, y], trajectory),
            iterations // sum(trajectory),
            Position.from_list([x, y], start),
        )

    def limit(
        self, trajectory, iterations, start=[1, 1], vector=Matrix([[0], [1]])
    ) -> sp.Float:
        return self.walk(trajectory, iterations, start).limit(vector)
