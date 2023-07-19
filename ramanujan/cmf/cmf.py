import sympy as sp
import pickle
from sympy.abc import n, x, y
from ramanujan import Matrix, Vector, simplify
from ramanujan.pcf import PCFFromMatrix


class CMF:
    r"""
    Represents a Conservative Matrix Field (CMF).

    A CMF is defined by two matrices $Mx, My$ that satisfy the perservation quality:
    $Mx(x, y) \cdot My(x+1, y) = My(x, y) \cdot Mx(x, y+1)$
    """

    def __init__(self, Mx: Matrix, My: Matrix, maximal_cache_dims=(50, 50), 
        initial_mat=Matrix(sp.eye(2)), initial_loc=(1,1), potential_cache_file=None):
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

        if potential_cache_file is not None:
            self.potential_cache = self.load_potential_from_file(potential_cache_file)
        else:
            self.potential_cache = [
                [None for _ in range(maximal_cache_dims[1]+1)] 
                    for _ in range(maximal_cache_dims[0]+1)]
        self.initial_loc = initial_loc
        self.potential_cache[initial_loc[0]][initial_loc[1]] = initial_mat

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
        m = self.Mx.walk({x: 1, y: 0}, trajectory[x], {x: x, y: y})
        m *= self.My.walk({x: 0, y: 1}, trajectory[y], {x: x + trajectory[x], y: y})
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
        self, trajectory: dict, iterations: int, start: dict = {x: 1, y: 1}
    ) -> Matrix:
        r"""
        Returns the multiplication matrix of walking in a certain trajectory.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: a dict containing the amount of steps in each direction.
            iterations: the amount of multiplications to perform
            start: a dict representing the starting point of the multiplication.
        Returns:
            the walk multiplication as defined above.
        """
        trajectory_matrix = self.trajectory_matrix(trajectory, start)
        return trajectory_matrix.walk(
            {n: 1},
            iterations // sum(trajectory.values()),
            {n: 1},
        )

    def limit(
        self,
        trajectory: dict,
        iterations: int,
        start: dict = {x: 1, y: 1},
        vector: Vector = Vector.zero(),
    ) -> sp.Float:
        """
        Returns the convergence limit of walking in a certain trajectory.

        This is essentially the same as `self.walk(trajectory, iterations, start).limit(vector)`

        Args:
            trajectory: a dict containing the amount of steps in each direction.
            iterations: the amount of multiplications to perform
            start: a dict representing the starting point of the multiplication.
            vector: The final vector to multiply the matrix by (the zero vector by default)
        Returns:
            the walk multiplication as defined above.
        """
        return self.walk(trajectory, iterations, start).limit(vector)

    def __getitem__(self, location):
        """
        Same as `potential`.
        """
        return self.potential(location)

    def potential(self, location):
        r"""
        Returns the potential matrix of the conservative matrix field at a given location.

        This function is not implemented using CMF.walk, to enable caching of CMF potential 
        cells calculated in the process.

        NOTICE - the cache is stored using python lists, and therefore begins at index 0.
        This might not the first index of the CMF, which will lead to None values in the 
        procedding potential cells.
        """
        x_target, y_target = location
        x_start, y_start = self.initial_loc

        if self.potential_cache[x_target][y_target] is not None:
            return self.potential_cache[x_target][y_target]

        # TODO - move this implementation to a decorator. Something like @cache.

        # We calculate the CMF potential at a location (x,y) by first multiplying the M_x
        # matrices, and then the M_y matrices. Using this convention, we can use the cached values
        # to make our calculation shorter.

        # Proceeding the x direction from the initial location - y is constant and equal to y_start
        for x_i in range(x_start, x_target+1):
            if self.potential_cache[x_i][y_start] is not None:
                continue

            self.potential_cache[x_i][y_start] = \
                self.potential_cache[x_i-1][y_start] * self.Mx.subs({x: x_i-1, y: y_start})

        # Proceeding the y direction from (x_target, y_start)
        for y_i in range(y_start, y_target+1):
            if self.potential_cache[x_target][y_i] is not None:
                continue

            self.potential_cache[x_target][y_i] = \
                self.potential_cache[x_target][y_i-1] * self.My.subs({x: x_target, y: y_i-1})

        return self.potential_cache[x_target][y_target]

    def calculate_over_range(self, max_x, max_y):
        """
        calculates the CMF in first quandrant up to (max_x, max_y).
        """
        
        # The calculation method in `potential` takes steps over the x axis first, then over y.
        # Since it caches elements while calculating, this will lead to caching of the entire space. 
        for x_i in range(self.initial_loc[0], max_x+1):
            self.potential([x_i, max_y])

    def dump_potential_to_file(self, dest_path):
        with open(dest_path, 'wb') as f:
            pickle.dump(self.potential_cache, f)

    def load_potential_from_file(self, cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
