from __future__ import annotations
import itertools
from multimethod import multimethod
from typing import Dict, List, Set, Union

import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix, Limit, simplify


class CMF:
    r"""
    Represents a Conservative Matrix Field (CMF).

    A CMF is defined by a set of axes and their relevant matrices that satisfy the perservation quality:
    for every two axes x and y, $Mx(x, y) \cdot My(x+1, y) = My(x, y) \cdot Mx(x, y+1)$.
    """

    def __init__(
        self,
        matrices: Dict[sp.Symbol, Matrix],
        validate: bool = True,
        axes_sorter=lambda axes, trajectory, position: sorted(axes, key=str),
    ):
        """
        Initializes a CMF with `Mx` and `My` matrices.

        Args:
            matrices: The CMF matrices as a dict,
                where the keys are the axes of the CMF and their values are the corresponding matrices.

        Raises:
            ValueError: if one of the matrices contain the symbol n, which is recserved for PCF conversions.
        """
        self.matrices = matrices
        if n in self.matrices.keys():
            raise ValueError(
                "Do not use symbol n as an axis, it's reserved for companion conversions"
            )
        self.axes_sorter = axes_sorter
        self.assert_matrices_same_dimension()
        if validate:
            self.assert_conserving()

    def __eq__(self, other) -> bool:
        return self.matrices == other.matrices

    def __repr__(self) -> str:
        return f"CMF({self.matrices})"

    def _are_conserving(
        self,
        x: sp.Symbol,
        y: sp.Symbol,
        x_forward: bool = True,
        y_forward: bool = True,
    ) -> bool:
        Mx = self.M(x, x_forward)
        My = self.M(y, y_forward)
        Mxy = simplify(Mx * My({x: x + 1 if x_forward else x - 1}))
        Myx = simplify(My * Mx({y: y + 1 if y_forward else y - 1}))
        return Mxy == Myx

    def assert_conserving(self, check_negatives: bool = False) -> None:
        """
        Asserts that all of the matrices of the CMF are conserving.
        Args:
            check_negatives: if `True`, will also check that the negative matrices are conserving.
                             this should mathematically always be the case when the positive matrices are conserving.
        Raises:
            ValueError: if two matrices or more are not conserving.
        """
        for x, y in itertools.combinations(self.matrices.keys(), 2):
            if not self._are_conserving(x, y, True, True):
                raise ValueError(f"M({x}) and M({y}) matrices are not conserving!")

            if check_negatives:
                if not self._are_conserving(x, y, False, True):
                    raise ValueError(f"M(-{x}) and M({y}) matrices are not conserving!")
                if not self._are_conserving(x, y, True, False):
                    raise ValueError(f"M({x}) and M(-{y}) matrices are not conserving!")
                if not self._are_conserving(x, y, False, False):
                    raise ValueError(
                        f"M(-{x}) and M(-{y}) matrices are not conserving!"
                    )

    def assert_matrices_same_dimension(self) -> None:
        """
        Asserts that all of the matrices of the CMF have the same dimensions.
        Raises:
            ValueError: in case the matrices are not conserving.
        """
        expected_N = self.N()
        for symbol, matrix in self.matrices.items():
            if not expected_N == matrix.rows and expected_N == matrix.cols:
                raise ValueError(
                    f"M({symbol}) is of dimension {matrix.rows}x{matrix.cols}, expected {expected_N}x{expected_N}"
                )

    def M(self, axis: sp.Symbol, sign: bool = True) -> Matrix:
        """
        Returns the axis matrix for a given axis.

        If sign is negative, returns the matrix corresponding a step back.
        Note that we do not reduce M because it might impair the conservative property.
        """
        if sign:
            return self.matrices[axis]
        else:
            return self.matrices[axis].inverse()({axis: axis - 1})

    def axes(self) -> Set[sp.Symbol]:
        """
        Returns the symbols of all axes of the CMF.
        """
        return set(self.matrices.keys())

    def axis_vector(self, axis: sp.Symbol, sign: bool = True) -> Dict[sp.Symbol, int]:
        """
        Given a CMF axis symbol `axis`,
        Returns the vector which is a single step in that axis and 0 in all other axes.
        The step is 1 with the sign of `sign`
        """
        step = 1 if sign else -1
        return {i: step if i == axis else 0 for i in self.axes()}

    def parameters(self) -> Set[sp.Symbol]:
        """
        Returns all (non-axis) symbolic parameters of the CMF.
        """
        return self.free_symbols() - self.axes()

    def free_symbols(self) -> Set[sp.Symbol]:
        """
        Returns all symbolic variables of the CMF, both axes and parameters.
        """
        return set.union(
            *list(map(lambda matrix: matrix.free_symbols, self.matrices.values()))
        )

    def dim(self) -> int:
        """
        Returns the dimension of the CMF,
        which is defined as the amount of axes of the CMF.
        """
        return len(self.axes())

    def N(self) -> int:
        """
        Returns the row/column amount of matrices of the CMF.
        """
        random_matrix = list(self.matrices.values())[0]
        return random_matrix.rows

    def subs(self, *args, **kwargs) -> CMF:
        """
        Returns a new CMF with substituted matrices.
        """
        return CMF(
            matrices={
                symbol: matrix.subs(*args, **kwargs)
                for symbol, matrix in self.matrices.items()
            },
            validate=False,
        )

    def simplify(self):
        """
        Returns a new CMF with simplified Mx and My
        """
        return CMF(
            matrices={
                symbol: simplify(matrix) for symbol, matrix in self.matrices.items()
            }
        )

    @staticmethod
    def decompose_trajectory(original_traj):
        iterations = []
        current_traj = original_traj.copy()

        while any(abs(value) > 0 for value in current_traj.values()):
            # Find the smallest positive integer value
            d = min(abs(value) for value in current_traj.values() if value > 0)

            # Create first_traj with non-zero values set to d
            first_traj = {
                key: int(sp.sign(value)) for key, value in current_traj.items()
            }

            # Create first_res with values subtracted by d
            first_res = {
                key: value - d * sp.sign(value) for key, value in current_traj.items()
            }

            # Store the result of this iteration
            iterations.append((d, first_traj, first_res))

            # Update current_traj for the next iteration
            current_traj = first_res
        return iterations

    def d_times_t_eval(self, eval_dict, traj_dict, d):
        from sympy.abc import m

        product = sp.eye(self.N())
        start = {
            axis: eval_dict[axis] + m * traj_dict[axis] for axis in eval_dict.keys()
        }
        A_m = self.walk(traj_dict, 1, start)
        # Iterate through eval_dicts, ensuring consistency with cmf.walk
        product = A_m.walk({m: 1}, int(d), {m: 0})
        return product

    def amazingTraj(self, start, traj):
        # do the usual amazing trajectory trick and get a stard point
        # frist brake traj T = d_1T_1+d_2T_2+d_3T_3+...
        decomposed_traj = CMF.decompose_trajectory(traj)
        # here is a list of (d,the_d_traj, left over traj)
        product = sp.eye(self.N())

        for d, T, left_over in decomposed_traj:
            product = product * CMF.d_times_t_eval(self, start, T, d).applyfunc(
                sp.factor
            )
            start = {axis: start[axis] + d * T[axis] for axis in start.keys()}

        return product.applyfunc(sp.factor)

    def trajectory_matrix(self, trajectory: dict, start: dict = None) -> Matrix:
        """
        Returns a corresponding matrix for walking in a trajectory, up to a constant.
        If `start` is given, the new matrix will be reduced to a single variable `n`.
        Args:
            trajectory: a dict containing the amount of steps in each direction.
            start: a dict representing the starting point of the multiplication.
        Returns:
            A matrix that represents a single step in the desired trajectory
        Raises:
            ValueError: if trajectory, start and matrix keys do not match.
        """
        if self.axes() != trajectory.keys():
            raise ValueError(
                f"Trajectory axes {trajectory.keys()} do not match CMF axes {self.axes()}"
            )

        if start and self.axes() != start.keys():
            raise ValueError(
                f"Start axes {start.keys()} do not match CMF axes {self.axes()}"
            )

        if start is None:
            start = {axis: axis for axis in self.axes()}
        else:
            start = CMF.variable_reduction_substitution(trajectory, start)
            return self.amazingTraj(start, trajectory)
        return self.walk(trajectory, 1, start).applyfunc(sp.factor)

    @staticmethod
    def variable_reduction_substitution(trajectory: dict, start: dict) -> Matrix:
        """
        Returns the substitution that reduces all CMF variables into one variable `n`.

        This transformation is possible only when the starting point is known.
        Each incrementation of the variable `n` represents a full step in `trajectory`.

        Args:
            trajectory_matrix: The matrix to reduce.
            trajectory: The trajectory that was used to create the trajectory matrix.
            start: The starting point from which the walk operation is to be calculated.
        Returns:
            A dict representing the above variable reduction substitution
        Raises:
            ValueError: if trajectory keys and start keys do not match
        """
        from sympy.abc import n

        if start.keys() != trajectory.keys():
            raise ValueError(
                f"Trajectory axes {trajectory.keys()} do not match start axes {start.keys()}"
            )

        return {
            axis: start[axis] + (n - 1) * trajectory[axis] for axis in trajectory.keys()
        }

    @multimethod
    def walk(  # noqa: F811
        self,
        trajectory: dict,
        iterations: List[int],
        start: dict,
    ) -> List[Matrix]:
        r"""
        Returns a list of trajectorial walk multiplication matrices in the desired depths.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: A dict containing the amount of steps in each direction.
            iterations: The amount of trajectory matrix multiplications to perform, either an integer or a list.
            start: A dict representing the starting point of the multiplication.
        Returns:
            The limit of the walk multiplication as defined above.
            If `iterations` is a list, returns a list of limits.
        """
        iterations_set = set(iterations)
        if len(iterations_set) != len(iterations):
            raise ValueError(f"`iterations` values must be unique, got {iterations}")

        if not iterations == sorted(iterations):
            raise ValueError(f"Iterations must be sorted, got {iterations}")

        if not all(depth >= 0 for depth in iterations):
            raise ValueError(
                f"iterations must contain only non-negative values, got {iterations}"
            )

        results = []
        position = start
        matrix = Matrix.eye(self.N())
        previous_depth = 0
        for depth in iterations:
            effective_depth = depth - previous_depth
            matrix *= self.walk(trajectory, effective_depth, position)
            position = {
                key: position[key] + value * effective_depth
                for key, value in trajectory.items()
            }
            previous_depth = depth
            results.append(matrix)
        return results

    @multimethod
    def walk(  # noqa: F811
        self,
        trajectory: dict,
        iterations: int,
        start: dict,
    ) -> Matrix:
        position = dict(start)
        matrix = Matrix.eye(self.N())
        for axis in self.axes_sorter(self.axes(), trajectory, position):
            depth = trajectory[axis] * iterations
            sign = depth >= 0
            matrix *= self.M(axis, sign).walk(
                self.axis_vector(axis, sign), abs(depth), position
            )
            position[axis] += depth
        return matrix

    @multimethod
    def limit(
        self,
        trajectory: dict,
        iterations: List[int],
        start: dict,
        p_vectors: Union[List[Matrix], None] = None,
        q_vectors: Union[List[Matrix], None] = None,
    ) -> List[Limit]:
        r"""
        Returns a list of limits of trajectorial walk multiplication matrices in the desired depths.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: A dict containing the amount of steps in each direction.
            iterations: The amount of trajectory matrix multiplications to perform, either an integer or a list.
            start: A dict representing the starting point of the multiplication.
        Returns:
            The limit of the walk multiplication as defined above.
            If `iterations` is a list, returns a list of limits.
        """

        def walk_function(iterations):
            return self.walk(trajectory, iterations, start)

        return Limit.walk_to_limit(iterations, walk_function, p_vectors, q_vectors)

    @multimethod
    def limit(  # noqa: F811
        self,
        trajectory: dict,
        iterations: int,
        start: dict,
        p_vectors: Union[List[Matrix], None] = None,
        q_vectors: Union[List[Matrix], None] = None,
    ) -> Limit:
        return self.limit(trajectory, [iterations], start, p_vectors, q_vectors)[0]

    def delta(
        self,
        trajectory: dict,
        depth: int,
        start: dict,
        limit: float = None,
        p_vectors: Union[List[Matrix], None] = None,
        q_vectors: Union[List[Matrix], None] = None,
    ):
        r"""
        Calculates the irrationality measure $\delta$ defined, as:
        $|\frac{p_n}{q_n} - L| = \frac{1}{q_n}^{1+\delta}$
        for the walk specified by the trajectory.
        $p_n$ and $q_n$ are respectively defined as the [0,-1] and [1,-1] elements in walk's matrix (see `Limit` class).

        If limit is not specified (i.e, limit is None),
        then limit is approximated as limit = self.limit(2 * depth)

        Args:
            trajectory: the trajectory of the walk.
            depth: $n$, is the number of trajectory matrices multiplied.
                The $\ell_1$ distance walked from the start point is `depth * sum(trajectory.values())`.
            start: the starting point of the walk operation.
            limit: $L$
            p_vectors: numerator extraction vectors for delta
            q_vectors: denominator extraction vectors for delta
        Returns:
            the delta value as defined above.
        """
        if limit is None:
            m, mlim = self.limit(
                trajectory, [depth, 2 * depth], start, p_vectors, q_vectors
            )
            limit = mlim.as_float()
        else:
            m = self.limit(trajectory, depth, start, p_vectors, q_vectors)
        return m.delta(limit)

    def delta_sequence(
        self,
        trajectory: dict,
        depth: int,
        start: dict,
        limit: float = None,
        p_vectors: Union[List[Matrix], None] = None,
        q_vectors: Union[List[Matrix], None] = None,
    ):
        r"""
        Calculates delta values sequentially up to `depth`.

        Args:
            trajectory: the trajectory of the walk.
            depth: $n$, is the number of trajectory matrices multiplied.
                The $\ell_1$ distance walked from the start point is `depth * sum(trajectory.values())`.
            start: the starting point of the walk operation.
            limit: $L$
            p_vectors: numerator extraction vectors for delta
            q_vectors: denominator extraction vectors for delta
        Returns:
            the delta value as defined above.
        """
        depths = list(range(1, depth + 1))
        if limit is None:
            depths += [2 * depth]
            approximants = self.limit(trajectory, depths, start)
            limit = approximants[-1].as_float()
            approximants = approximants[:-1]
        else:
            approximants = self.limit(trajectory, depths, start)
        return [approximant.delta(limit) for approximant in approximants]
