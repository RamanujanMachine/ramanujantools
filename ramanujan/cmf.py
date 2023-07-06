import sympy as sp
from sympy.abc import n, x, y

from ramanujan import Matrix, Position, simplify


class CMF:
    variables = [x, y]

    @staticmethod
    def _initialize_positions(*positions):
        return (CMF._initialize_position(position) for position in positions)

    @staticmethod
    def _initialize_position(position):
        if type(position) == list:
            return dict(zip(CMF.variables, position))
        return position

    def __init__(self, Mx: Matrix, My: Matrix):
        self.m_Mx = Mx
        self.m_My = My
        Mxy = simplify(self.Mx(x, y) * self.My(x + 1, y))
        Myx = simplify(self.My(x, y) * self.Mx(x, y + 1))
        if simplify(Mxy - Myx) != Matrix([[0, 0], [0, 0]]):
            raise ValueError("The given Mx and My matrices are not conserving!")

    def Mx(self, _x, _y) -> Matrix:
        """Substitute Mx."""
        return self.m_Mx.subs({x: _x, y: _y})

    def My(self, _x, _y) -> Matrix:
        """Substitute My."""
        return self.m_My.subs({x: _x, y: _y})

    def subs(self, substitutions):
        """Returns a new CMF with substituted Mx and My."""
        return CMF(self.m_Mx.subs(substitutions), self.m_My.subs(substitutions))

    def trajectory_matrix(self, trajectory, start=None) -> Matrix:
        """Returns the corresponding matrix for walking in trajectory.

        If start is given, the new matrix will be reduces to a single variable `n`.
        """
        start, trajectory = CMF._initialize_positions(start, trajectory)
        m = self.m_Mx.walk({x: 1, y: 0}, trajectory[x], {x: x, y: y})
        m *= self.m_My.walk({x: 0, y: 1}, trajectory[y], {x: x + trajectory[x], y: y})
        if start is not None:
            m = CMF.substitute_trajectory(m, trajectory, start)
        return simplify(m)

    @staticmethod
    def substitute_trajectory(trajectory_matrix: Matrix, trajectory, start):
        """Returns trajectory_matrix reduced to a single variable `n`."""
        from sympy.abc import n

        start, trajectory = CMF._initialize_positions(start, trajectory)
        sub = lambda i: start[i] + (n - 1) * trajectory[i]
        return trajectory_matrix.subs([(x, sub(x)), (y, sub(y))])

    def walk(self, trajectory, iterations, start=[1, 1]) -> Matrix:
        """Returns the multiplication matrix of walking in a certain trajectory."""
        trajectory_matrix = self.trajectory_matrix(trajectory, start)
        return trajectory_matrix.walk(
            {n: 1},
            iterations // sum(trajectory),
            {n: 1},
        )

    def limit(
        self, trajectory, iterations, start=[1, 1], vector=Matrix([[0], [1]])
    ) -> sp.Float:
        """Returns the limit of walking in a certain trajectory."""
        return self.walk(trajectory, iterations, start).limit(vector)
