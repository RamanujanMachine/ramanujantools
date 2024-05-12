from __future__ import annotations
import itertools
from multimethod import multimethod
from typing import Collection, Dict, List, Set, Union

import sympy as sp
from sympy.abc import n

from ramanujan import SquareMatrix, Limit, simplify
from ramanujan.pcf import PCFFromSquareMatrix


class CMF:
    r"""
    Represents a Conservative SquareMatrix Field (CMF).

    A CMF is defined by a set of axes and their relevant matrices that satisfy the perservation quality:
    for every two axes x and y, $Mx(x, y) \cdot My(x+1, y) = My(x, y) \cdot Mx(x, y+1)$
    """

    def __init__(self, matrices: Dict[sp.Symbol, SquareMatrix]):
        """
        Initializes a CMF with `Mx` and `My` matrices
        """
        self.matrices = matrices
        self.assert_matrices_same_dimension()
        self.assert_conserving()
        assert (
            n not in self.matrices.keys()
        ), "Do not use symbol n as an axis, it's reserved for PCF conversions"

    def __eq__(self, other) -> bool:
        return self.matrices == other.matrices

    def __repr__(self) -> str:
        return f"CMF({self.matrices})"

    def are_conserving(
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
            AssertionError: in case the matrices are not conserving.
        """
        for x, y in itertools.combinations(self.matrices.keys(), 2):
            if not self.are_conserving(x, y, True, True):
                raise ValueError(f"M({x}) and M({y}) matrices are not conserving!")

            if check_negatives:
                if not self.are_conserving(x, y, False, True):
                    raise ValueError(f"M(-{x}) and M({y}) matrices are not conserving!")
                if not self.are_conserving(x, y, True, False):
                    raise ValueError(f"M({x}) and M(-{y}) matrices are not conserving!")
                if not self.are_conserving(x, y, False, False):
                    raise ValueError(
                        f"M(-{x}) and M(-{y}) matrices are not conserving!"
                    )

    def assert_matrices_same_dimension(self) -> None:
        """
        Asserts that all of the matrices of the CMF have the same dimensions.
        Raises:
            AssertionError: in case the matrices are not conserving.
        """
        matrices_dimensions = set(map(lambda m: m.N(), self.matrices.values()))
        assert (
            len(matrices_dimensions) == 1
        ), f"Received matrices of different dimensions: {matrices_dimensions}"
        self.assert_conserving()

    def M(self, axis: sp.Symbol, sign: bool = True) -> SquareMatrix:
        """
        Returns the axis matrix for a given axis.

        If sign is negative, returns the matrix corresponding a step back.
        Note that we do not normalize M because it might impair the conservative property.
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
        return random_matrix.N()

    def subs(self, *args, **kwargs) -> CMF:
        """Returns a new CMF with substituted Mx and My."""
        return CMF(
            matrices={
                symbol: matrix.subs(*args, **kwargs)
                for symbol, matrix in self.matrices.items()
            }
        )

    def simplify(self):
        """Returns a new CMF with simplified Mx and My"""
        return CMF(
            matrices={symbol: simplify(matrix) for symbol, matrix in self.matrices}
        )

    def default_origin(self):
        """
        Returns the default origin value, which is 1 for every axis.
        """
        return {axis: 1 for axis in self.axes()}

    def trajectory_matrix(self, trajectory: dict, start: dict = None) -> SquareMatrix:
        """

        Returns the corresponding matrix for walking in trajectory.
        If `start` is given, the new matrix will be reduced to a single variable `n`.
        Args:
            trajectory: a dict containing the amount of steps in each direction.
            start: a dict representing the starting point of the multiplication.
        Returns:
            A matrix that represents a single step in the desired trajectory
        """
        assert (
            self.axes() == trajectory.keys()
        ), f"Trajectory axes {trajectory.keys()} does not match CMF axes {self.axes()}"

        if start:
            assert (
                self.axes() == start.keys()
            ), f"Start axes {start.keys()} does not match CMF axes {self.axes()}"

        position = {axis: axis for axis in self.axes()}
        m = sp.eye(2)
        for axis in self.axes():
            sign = trajectory[axis] >= 0
            m *= self.M(axis, sign).walk(
                self.axis_vector(axis, sign), abs(trajectory[axis]), position
            )
            position[axis] += trajectory[axis]
        if start:
            m = CMF.substitute_trajectory(m, trajectory, start)
        return m.normalize()

    def as_pcf(self, trajectory) -> PCFFromSquareMatrix:
        """
        Returns the PCF equivalent to the CMF in a certain trajectory, up to a mobius transform.
        """
        return self.trajectory_matrix(trajectory, self.default_origin()).as_pcf()

    @staticmethod
    def substitute_trajectory(
        trajectory_matrix: SquareMatrix, trajectory: dict, start: dict
    ) -> SquareMatrix:
        """
        Returns trajectory_matrix reduced to a single variable `n`.

        This transformation is possible only when the starting point is known.
        Each incrementation of the variable `n` represents a full step in `trajectory`.
        """
        from sympy.abc import n

        assert (
            trajectory.keys() == start.keys()
        ), f"Key mismatch between trajectory ({trajectory.keys()}) and start ({start.keys()})"

        def sub(i):
            return start[i] + (n - 1) * trajectory[i]

        return trajectory_matrix.subs([(axis, sub(axis)) for axis in trajectory.keys()])

    @multimethod
    def walk(
        self,
        trajectory: dict,
        iterations: Collection[int],
        start: Union[dict, type(None)] = None,
    ) -> List[SquareMatrix]:
        r"""
        Returns a list of trajectorial walk multiplication matrices in the desired depths.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: a dict containing the amount of steps in each direction.
            iterations: the amount of multiplications to perform, either an integer or a list.
            start: a dict representing the starting point of the multiplication, `default_origin` by default.
        Returns:
            The limit of the walk multiplication as defined above.
            If `iterations` is a list, returns a list of limits.
        """
        trajectory_matrix = self.trajectory_matrix(
            trajectory, start or self.default_origin()
        )
        actual_iterations = [i // sum(trajectory.values()) for i in iterations]
        return trajectory_matrix.walk({n: 1}, actual_iterations, {n: 1})

    @multimethod
    def walk(  # noqa: F811
        self,
        trajectory: dict,
        iterations: int,
        start: Union[dict, type(None)] = None,
    ) -> SquareMatrix:
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
            trajectory: a dict containing the amount of steps in each direction.
            iterations: the amount of multiplications to perform, either an integer or a list.
            start: a dict representing the starting point of the multiplication, `default_origin` by default.
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
