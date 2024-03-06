from sympy.abc import n, x, y

from typing import List

from ramanujan import Matrix, simplify, zero
from ramanujan.pcf import PCFFromMatrix


class CMF:
    r"""
    Represents a Conservative Matrix Field (CMF).

    A CMF is defined by two matrices $Mx, My$ that satisfy the perservation quality:
    $Mx(x, y) \cdot My(x+1, y) = My(x, y) \cdot Mx(x, y+1)$
    """

    def __init__(self, Mx: Matrix, My: Matrix):
        """
        Initializes a CMF with `Mx` and `My` matrices
        """

        self.Mx = Mx
        """The Mx matrix of the CMF"""
        self.My = My
        """The My matrix of the CMF"""

        Mxy = simplify(self.Mx * self.My({x: x + 1}))
        Myx = simplify(self.My * self.Mx({y: y + 1}))
        if simplify(Mxy - Myx) != Matrix([[0, 0], [0, 0]]):
            raise ValueError("The given Mx and My matrices are not conserving!")

    def __eq__(self, other):
        return self.Mx == other.Mx and self.My == other.My

    def __repr__(self):
        return f"CMF({self.Mx}, {self.My})"

    def subs(self, *args, **kwrags):
        """Returns a new CMF with substituted Mx and My."""
        return CMF(self.Mx.subs(*args, **kwrags), self.My.subs(*args, **kwrags))

    def simplify(self):
        """Returns a new CMF with simplified Mx and My"""
        return CMF(simplify(self.Mx), simplify(self.My))

    def trajectory_matrix(self, trajectory: dict, start: dict = None) -> Matrix:
        """
        Returns the corresponding matrix for walking in trajectory.

        If `start` is given, the new matrix will be reduced to a single variable `n`.
        Args:
            trajectory: a dict containing the amount of steps in each direction.
            start: a dict representing the starting point of the multiplication.
        Returns:
            A matrix that represents a single step in the desired trajectory
        """
        m = self.Mx.walk({x: 1, y: 0}, trajectory[x], {x: x, y: y})[-1]
        m *= self.My.walk({x: 0, y: 1}, trajectory[y], {x: x + trajectory[x], y: y})[-1]
        if start is not None:
            m = CMF.substitute_trajectory(m, trajectory, start)
        return simplify(m)

    def as_pcf(self, trajectory, start: dict = {x: 1, y: 1}) -> PCFFromMatrix:
        """
        Returns the PCF equivalent to the CMF in a certain trajectory, up to a mobius transform.
        """
        return self.trajectory_matrix(trajectory, start).as_pcf()

    @staticmethod
    def substitute_trajectory(
        trajectory_matrix: Matrix, trajectory: dict, start: dict
    ) -> Matrix:
        """
        Returns trajectory_matrix reduced to a single variable `n`.

        This transformation is possible only when the starting point is known.
        Each incrementation of the variable `n` represents a full step in `trajectory`.
        """
        from sympy.abc import n

        def sub(i):
            return start[i] + (n - 1) * trajectory[i]

        return trajectory_matrix.subs([(x, sub(x)), (y, sub(y))])

    def walk(
        self, trajectory: dict, iterations: int, start: dict = {x: 1, y: 1},scoops = set()
        ) -> List[Matrix]:
        r"""
        Returns the multiplication matrix(ces) of walking in a certain trajectory.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: a dict containing the amount of steps in each direction.
            iterations: the amount of multiplications to perform
            start: a dict representing the starting point of the multiplication.
        Returns:
            Steps from the walk multiplication as defined above.
        """
        trajectory_matrix = self.trajectory_matrix(trajectory, start)
        return trajectory_matrix.walk(
            {n: 1},
            iterations // sum(trajectory.values()),
            {n: 1},
            scoops = scoops
        )

    def limit(
        self,
        trajectory: dict,
        iterations: int,
        start: dict = {x: 1, y: 1},
        vector: Matrix = zero(),
    ) -> Matrix:
        """
        Returns the convergence limit of walking in a certain trajectory.

        This is essentially the same as `self.walk(trajectory, iterations, start)[-1] * vector`

        Args:
            trajectory: a dict containing the amount of steps in each direction.
            iterations: the amount of multiplications to perform
            start: a dict representing the starting point of the multiplication.
            vector: The final vector to multiply the matrix by (the zero vector by default)
        Returns:
            the limit of the walk multiplication as defined above.
        """
        return self.walk(trajectory, iterations, start)[-1] * vector
