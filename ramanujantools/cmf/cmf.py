from __future__ import annotations
import itertools
from multimethod import multimethod
from typing import Collection, Dict, List, Set, Union

import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix, Limit, simplify
from ramanujantools.pcf import PCFFromMatrix


class CMF:
    r"""
    Represents a Conservative Matrix Field (CMF).

    A CMF is defined by a set of axes and their relevant matrices that satisfy the perservation quality:
    for every two axes x and y, $Mx(x, y) \cdot My(x+1, y) = My(x, y) \cdot Mx(x, y+1)$.
    """

    def __init__(self, matrices: Dict[sp.Symbol, Matrix], validate=True):
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
                "Do not use symbol n as an axis, it's reserved for PCF conversions"
            )
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
        Mxy = simplify(Mx * My.subs({x: x + 1 if x_forward else x - 1}))
        Myx = simplify(My * Mx.subs({y: y + 1 if y_forward else y - 1}))
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
            return self.matrices[axis].inverse().subs({axis: axis - 1})

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
        Returns a new CMF with substituted Mx and My.
        """
        return CMF(
            matrices={
                symbol: matrix.subs(*args, **kwargs)
                for symbol, matrix in self.matrices.items()
            }
        )

    def simplify(self):
        """
        Returns a new CMF with simplified Mx and My
        """
        return CMF(
            matrices={symbol: simplify(matrix) for symbol, matrix in self.matrices}
        )

    def default_origin(self):
        """
        Returns the default origin value, which is 1 for every axis.
        """
        return {axis: 1 for axis in self.axes()}

    def trajectory_matrix(self, trajectory: dict, start: dict = None) -> Matrix:
        """

        Returns the corresponding matrix for walking in trajectory.
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

        position = {axis: axis for axis in self.axes()}
        m = sp.eye(self.N())
        for axis in self.axes():
            sign = trajectory[axis] >= 0
            m *= self.M(axis, sign).walk(
                self.axis_vector(axis, sign), abs(trajectory[axis]), position
            )
            position[axis] += trajectory[axis]
        if start:
            m = CMF.substitute_trajectory(m, trajectory, start)
        return m.reduce()

    def as_pcf(self, trajectory) -> PCFFromMatrix:
        """
        Returns the PCF equivalent to the CMF in a certain trajectory, up to a mobius transform.
        """
        return self.trajectory_matrix(trajectory, self.default_origin()).as_pcf()

    @staticmethod
    def substitute_trajectory(
        trajectory_matrix: Matrix, trajectory: dict, start: dict
    ) -> Matrix:
        """
        Reduces a trajectory matrix to have a single variable `n`.

        This transformation is possible only when the starting point is known.
        Each incrementation of the variable `n` represents a full step in `trajectory`.

        Args:
            trajectory_matrix: The matrix to reduce.
            trajectory: The trajectory that was used to create the trajectory matrix.
            start: The starting point from which the walk operation is to be calculated.
        Returns:
            A matrix that with one variable n that is equivalent to trajectory matrix,
            such that every step in the n axis is eqivalent to a step in `trajectory` when starting from `start`.
        Raises:
            ValueError: if trajectory keys and start keys do not match
        """
        from sympy.abc import n

        if start.keys() != trajectory.keys():
            raise ValueError(
                f"Trajectory axes {trajectory.keys()} do not match start axes {start.keys()}"
            )

        def sub(i):
            return start[i] + (n - 1) * trajectory[i]

        return trajectory_matrix.subs([(axis, sub(axis)) for axis in trajectory.keys()])

    @multimethod
    def walk(
        self,
        trajectory: dict,
        iterations: Collection[int],
        start: Union[dict, type(None)] = None,
    ) -> List[Matrix]:
        r"""
        Returns a list of trajectorial walk multiplication matrices in the desired depths.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: A dict containing the amount of steps in each direction.
            iterations: The amount of trajectory matrix multiplications to perform, either an integer or a list.
            start: A dict representing the starting point of the multiplication, `default_origin` by default.
        Returns:
            The limit of the walk multiplication as defined above.
            If `iterations` is a list, returns a list of limits.
        """
        trajectory_matrix = self.trajectory_matrix(
            trajectory, start or self.default_origin()
        )
        return trajectory_matrix.walk({n: 1}, iterations, {n: 1})

    @multimethod
    def walk(  # noqa: F811
        self,
        trajectory: dict,
        iterations: int,
        start: Union[dict, type(None)] = None,
    ) -> Matrix:
        return self.walk(trajectory, [iterations], start)[0]

    @multimethod
    def limit(
        self,
        trajectory: dict,
        iterations: Collection[int],
        start: Union[dict, type(None)] = None,
    ) -> List[Limit]:
        r"""
        Returns a list of limits of trajectorial walk multiplication matrices in the desired depths.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: A dict containing the amount of steps in each direction.
            iterations: The amount of trajectory matrix multiplications to perform, either an integer or a list.
            start: A dict representing the starting point of the multiplication, `default_origin` by default.
        Returns:
            The limit of the walk multiplication as defined above.
            If `iterations` is a list, returns a list of limits.
        """
        return list(
            map(lambda matrix: Limit(matrix), self.walk(trajectory, iterations, start))
        )

    @multimethod
    def limit(  # noqa: F811
        self,
        trajectory: dict,
        iterations: int,
        start: Union[dict, type(None)] = None,
    ) -> Limit:
        return self.limit(trajectory, [iterations], start)[0]

    def delta(
            self,
            trajectory: dict,
            depth: int,
            start: dict = None,
            limit: float = None):
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
            limit: $L$
        Returns:
            the delta value as defined above.
        """
        if limit is None:
            depths = [depth, 2 * depth]
            approximants = self.limit(trajectory, depths, start)
            limit = approximants[-1].as_float()
            approximants = approximants[:-1]
        else:
            approximants = self.limit(trajectory, [depth], start)
        return approximants[0].delta(limit)

    def delta_sequence(
            self,
            trajectory: dict,
            depth: int,
            start: dict = None,
            limit: float = None):
        r"""
        Add description here
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
