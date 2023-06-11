import sympy as sp
from sympy.abc import n, x, y

from ramanujan import Matrix, simplify


class CMF:
    variables = [x, y]

    @staticmethod
    def _initialize_positions(*positions):
        return tuple(map(CMF._initialize_position, positions))

    @staticmethod
    def _initialize_position(position):
        if type(position) == list:
            return dict(zip(CMF.variables, position))
        return position

    def __init__(self, Mx: Matrix, My: Matrix):
        self.Mx = Mx
        self.My = My
        Mxy = simplify(self.Mx * self.My({x: x + 1}))
        Myx = simplify(self.My * self.Mx({y: y + 1}))
        if simplify(Mxy - Myx) != Matrix([[0, 0], [0, 0]]):
            raise ValueError("The given Mx and My matrices are not conserving!")

    def subs(self, substitutions):
        """Returns a new CMF with substituted Mx and My."""
        return CMF(self.Mx.subs(substitutions), self.My.subs(substitutions))

    def trajectory_matrix(self, trajectory, start=None) -> Matrix:
        """Returns the corresponding matrix for walking in trajectory.

        If start is given, the new matrix will be reduces to a single variable `n`.
        """
        trajectory, start = CMF._initialize_positions(trajectory, start)
        m = self.Mx.walk({x: 1, y: 0}, trajectory[x], {x: x, y: y})
        m *= self.My.walk({x: 0, y: 1}, trajectory[y], {x: x + trajectory[x], y: y})
        if start is not None:
            m = CMF.substitute_trajectory(m, trajectory, start)
        return simplify(m)

    @staticmethod
    def substitute_trajectory(trajectory_matrix: Matrix, trajectory, start):
        """Returns trajectory_matrix reduced to a single variable `n`."""
        from sympy.abc import n

        trajectory, start = CMF._initialize_positions(trajectory, start)

        def sub(i):
            return start[i] + (n - 1) * trajectory[i]

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
