from __future__ import annotations
from functools import lru_cache

import itertools

import sympy as sp
from sympy.abc import n

from ramanujantools import Position, Matrix, Limit, simplify
from ramanujantools.flint_core import (
    NumericMatrix,
    SymbolicMatrix,
    FlintContext,
    flint_ctx,
)

from ramanujantools.utils import batched, Batchable


class CMF:
    r"""
    Represents a Conservative Matrix Field (CMF).

    A CMF is defined by a set of axes and their relevant matrices that satisfy the perservation quality:
    for every two axes x and y, $Mx(x, y) \cdot My(x+1, y) = My(x, y) \cdot Mx(x, y+1)$.
    """

    def __init__(
        self,
        matrices: dict[sp.Symbol, Matrix],
        validate: bool = True,
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
        self.assert_matrices_same_dimension()
        if validate:
            self.assert_conserving()

    def __hash__(self) -> int:
        return hash(frozenset(self.matrices.items()))

    def __eq__(self, other) -> bool:
        return self.matrices == other.matrices

    def __repr__(self) -> str:
        return f"CMF({self.matrices})"

    def _latex(self, printer) -> str:
        lines = []
        for axis in sorted(self.axes(), key=str):
            lines.append(
                f"{printer.doprint(axis)} \\mapsto {printer.doprint(self.M(axis))}"
            )
        return r"$$\begin{array}{l}" + r"\\ ".join(lines) + r"\end{array}$$"

    def _repr_latex_(self) -> str:
        return rf"$${sp.latex(self)}$$"

    def __getstate__(self):
        return self.matrices

    def __setstate__(self, state):
        self.matrices = state

    def _are_conserving(
        self,
        x: sp.Symbol,
        y: sp.Symbol,
        x_forward: bool = True,
        y_forward: bool = True,
    ) -> bool:
        Mx = self.M(x, x_forward)
        My = self.M(y, y_forward)
        Mxy = (Mx * My({x: x + 1 if x_forward else x - 1})).factor()
        Myx = (My * Mx({y: y + 1 if y_forward else y - 1})).factor()
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

    def dual(self) -> CMF:
        """
        Returns the dual CMF which is defined with inverse-transpose matrices.
        """
        return CMF(
            {axis: self.M(axis).inverse().transpose() for axis in self.axes()},
            validate=False,
        )

    def axes(self) -> set[sp.Symbol]:
        """
        Returns the symbols of all axes of the CMF.
        """
        return set(self.matrices.keys())

    def parameters(self) -> set[sp.Symbol]:
        """
        Returns all (non-axis) symbolic parameters of the CMF.
        """
        return self.free_symbols() - self.axes()

    def free_symbols(self) -> set[sp.Symbol]:
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

    def _validate_axes_substitutions(self, substitutions: Position) -> None:
        if (
            axes_substitutions := self.axes().intersection(substitutions.keys())
            != set()
        ):
            raise ValueError(
                f"Cannot substitute axis parameters! got: {axes_substitutions}"
            )

    def subs(self, substitutions: Position) -> CMF:
        """
        Returns a new CMF with substituted matrices.
        """
        self._validate_axes_substitutions(substitutions)
        return CMF(
            matrices={
                symbol: matrix.subs(substitutions)
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
    def walk_symbol() -> sp.Symbol:
        return sp.Symbol("walk")

    def ctx(self, start: Position | None) -> FlintContext:
        start = Position(start) if start else Position()
        free_symbols = (
            self.free_symbols().union({CMF.walk_symbol()}).union(start.free_symbols())
        )
        return flint_ctx(free_symbols, fmpz=start.is_polynomial())

    def _calculate_diagonal_matrix_backtrack(
        self, trajectory: Position, start: Position, ctx: FlintContext
    ) -> SymbolicMatrix:
        """
        Inner function of an inner function. DO NOT USE DIRECTLY.
        This is the backtracking hook used for `_calculate_diagonal_marix`.
        It's used to look for a non-singular path towards the trajectroy matrix.
        """
        if trajectory.longest() == 0:
            return SymbolicMatrix.eye(self.N(), ctx)
        for axis in sorted(trajectory.keys(), key=str):
            try:
                inner_trajectory = trajectory.copy()
                position = start.copy()
                sign = inner_trajectory[axis] >= 0
                current = SymbolicMatrix.from_sympy(self.M(axis, sign), ctx).subs(
                    position
                )
                position[axis] += inner_trajectory.pop(axis)
                return current * self._calculate_diagonal_matrix_backtrack(
                    inner_trajectory, position, ctx
                )
            except ZeroDivisionError:
                continue
        raise ZeroDivisionError(
            "A singularity has occured in every possible trajectory combination"
        )

    @lru_cache
    def _calculate_diagonal_matrix(
        self, trajectory: Position, start: Position, ctx: FlintContext
    ) -> SymbolicMatrix:
        """
        The manual calculation of trajectory matrix in the stopping condition.
        You should probably use `trajectory_matrix` instead.

        Assumes trajectory is a simple diagonal - all abs values are at most 1
        Args:
            trajectory: a dict containing the amount of steps in each direction.
            start: a dict representing the starting point of the multiplication.
        Returns:
            A matrix that represents a single step in the desired trajectory
        """
        if trajectory.longest() > 1:
            raise ValueError(
                f"Called _calculate_diagonal_matrix with a trajectory that is not a simple diagonal: {trajectory}"
            )
        trajectory = Position(
            {axis: value for axis, value in trajectory.items() if value != 0}
        )

        return self._calculate_diagonal_matrix_backtrack(trajectory, start, ctx)

    def _work_symbolic(
        self,
        trajectory: Position,
        start: Position,
    ) -> SymbolicMatrix:
        """
        Internal work logic. Do not use directly.
        """
        ctx = self.ctx(start)
        # Stopping condition: l-infinity norm of trajectory is less than 1
        if trajectory.longest() <= 1:
            return self._calculate_diagonal_matrix(trajectory, start, ctx)

        result = SymbolicMatrix.eye(self.N(), ctx)
        symbol = CMF.walk_symbol()
        depth = trajectory.shortest()
        position = start.copy()
        while depth > 0:
            diagonal = trajectory.signs()
            result *= self._trajectory_matrix_inner(diagonal, position, symbol).walk(
                {symbol: 1}, int(depth), {symbol: 0}
            )
            position += depth * diagonal
            trajectory -= depth * diagonal
            depth = trajectory.shortest()
        return result

    def _work_numeric(
        self,
        trajectory: Position,
        start: Position,
    ) -> NumericMatrix:
        """
        Internal work logic. Do not use directly.
        """
        ctx = self.ctx(start)
        # Stopping condition: l-infinity norm of trajectory is less than 1
        if trajectory.longest() <= 1:
            matrix = self._calculate_diagonal_matrix(trajectory, start, ctx).factor()
            return NumericMatrix.lambda_from_rt(matrix)(start)

        result = NumericMatrix.eye(self.N())
        symbol = CMF.walk_symbol()
        depth = trajectory.shortest()
        position = start.copy()
        while depth > 0:
            diagonal = trajectory.signs()
            result *= NumericMatrix.walk(
                self.trajectory_matrix(diagonal, position, symbol),
                Position({symbol: 1}),
                int(depth),
                Position({symbol: 0}),
            )
            position += depth * diagonal
            trajectory -= depth * diagonal
            depth = trajectory.shortest()
        return result

    def work(self, start: Position, end: Position) -> Matrix:
        """
        Returns the transformation matrix corresponding to walking from start to end
        """
        start = Position(start)
        end = Position(end)
        trajectory = end - start
        if not trajectory.is_integer():
            raise ValueError(
                f"Can only calculate work for two points with integer trajectory between them! "
                f"start={start}, end={end}, trajectory={trajectory}"
            )
        if start.free_symbols() == set():
            return self._work_numeric(trajectory, start).to_rt()
        else:
            return self._work_symbolic(trajectory, start).factor()

    def _trajectory_matrix_inner(
        self, trajectory: Position, start: Position, symbol: sp.Symbol
    ) -> SymbolicMatrix:
        start = self.trajectory_substitution(trajectory, start, symbol)
        return self._work_symbolic(trajectory, start)

    def trajectory_matrix(
        self, trajectory: dict, start: dict, symbol: sp.Symbol = n
    ) -> Matrix:
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

        trajectory = Position(trajectory)
        start = Position(start)

        if self.axes() != trajectory.keys():
            raise ValueError(
                f"Trajectory axes {trajectory.keys()} do not match CMF axes {self.axes()}"
            )

        if not set(start.keys()).issubset(self.axes()):
            raise ValueError(
                f"Start axes {start.keys()} are not a subset of CMF axes {self.axes()}!"
            )

        return self._trajectory_matrix_inner(trajectory, start, symbol).factor()

    def trajectory_substitution(
        self, trajectory: Position, start: Position, symbol: sp.Symbol
    ) -> Position:
        """
        Returns the substitution of the start position to a function of the new trajectory symbol.

        This transformation is possible only when the starting point is known.
        Each incrementation of the variable `symbol` represents a full step in `trajectory`.

        Args:
            trajectory: The trajectory that was used to create the trajectory matrix.
            start: The starting point from which the walk operation is to be calculated.
            symbol: The new symbol of the reduced trajectory matrix.
        Returns:
            A dict representing the above variable reduction substitution
        Raises:
            ValueError: if trajectory keys and start keys do not match
        """
        trajectory = Position(trajectory)
        start = Position(start)

        if symbol in start or symbol in trajectory or symbol in start.free_symbols():
            raise ValueError(
                f"Attempted to run a trajectory_matrix where desired symbol is in start or trajectory! "
                f"symbol={symbol}, start={start}, trajectory={trajectory}"
            )
        effective_start = Position({axis: axis for axis in self.axes()})
        for axis in start:
            effective_start[axis] = start[axis]

        return effective_start + symbol * trajectory

    @batched("iterations")
    def walk(  # noqa: F811
        self,
        trajectory: dict,
        iterations: Batchable[int],
        start: dict,
    ) -> Batchable[Matrix]:
        r"""
        Returns a list of trajectorial walk multiplication matrices in the desired depths.

        The walk operation is defined as $\prod_{i=0}^{n-1}M_{s, t}(n)$,
        where M is `trajectory_matrix(trajectory, start)`, s is `start` and t is `trajectory`.

        Args:
            trajectory: A dict containing the amount of steps in each direction.
            iterations: The amount of trajectory matrix multiplications to perform, either an integer or a list.
            start: A dict representing the starting point of the multiplication.
        Returns:
            The limit of the walk multiplication as defined above.
            If `iterations` is a list, returns a list of limits.
        """
        if self.axes() != trajectory.keys():
            raise ValueError(
                f"Trajectory axes {trajectory.keys()} do not match CMF axes {self.axes()}"
            )

        if start and self.axes() != start.keys():
            raise ValueError(
                f"Start axes {start.keys()} do not match CMF axes {self.axes()}"
            )

        iterations_set = set(iterations)
        if len(iterations_set) != len(iterations):
            raise ValueError(f"`iterations` values must be unique, got {iterations}")

        if not iterations == sorted(iterations):
            raise ValueError(f"Iterations must be sorted, got {iterations}")

        if not all(depth >= 0 for depth in iterations):
            raise ValueError(
                f"iterations must contain only non-negative values, got {iterations}"
            )

        trajectory = Position(trajectory)
        start = Position(start)
        return self.trajectory_matrix(trajectory, start).walk(
            {n: 1}, iterations, {n: 0}
        )

    @batched("iterations")
    def limit(
        self,
        trajectory: dict,
        iterations: Batchable[int],
        start: dict,
        initial_values: Matrix | None = None,
        final_projection: Matrix | None = None,
    ) -> Batchable[Limit]:
        r"""
        Returns a list of limits of trajectorial walk multiplication matrices in the desired depths.

        The walk operation is defined as $\prod_{i=0}^{n-1}M(s_0 + i \cdot t_0, ..., s_k + i \cdot t_k)$,
        where `M=trajectory_matrix(trajectory, start)`, and `n / size(trajectory)` (L1 size - total amount of steps)

        Args:
            trajectory: A dict containing the amount of steps in each direction.
            iterations: The amount of trajectory matrix multiplications to perform, either an integer or a list.
            start: A dict representing the starting point of the multiplication.
            initial_values: the initial values matrix for the limit calculation
            final_projection: the final projection matrix for the limit calculation
        Returns:
            The limit of the walk multiplication as defined above.
            If `iterations` is a list, returns a list of limits.
        """

        def walk_function(iterations):
            return self.walk(trajectory, iterations, start)

        return Limit.walk_to_limit(
            iterations, walk_function, initial_values, final_projection
        )

    def delta(
        self,
        trajectory: dict,
        depth: int,
        start: dict,
        limit: float = None,
        initial_values: Matrix | None = None,
        final_projection: Matrix | None = None,
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
            initial_values: the initial values matrix for the limit calculation
            final_projection: the final projection matrix for the limit calculation
        Returns:
            the delta value as defined above.
        """
        if limit is None:
            m, mlim = self.limit(
                trajectory, [depth, 2 * depth], start, initial_values, final_projection
            )
            limit = mlim.as_float()
        else:
            m = self.limit(trajectory, depth, start, initial_values, final_projection)
        return m.delta(limit)

    def delta_sequence(
        self,
        trajectory: dict,
        depth: int,
        start: dict,
        limit: float = None,
        initial_values: Matrix | None = None,
        final_projection: Matrix | None = None,
    ):
        r"""
        Calculates delta values sequentially up to `depth`.

        Args:
            trajectory: the trajectory of the walk.
            depth: $n$, is the number of trajectory matrices multiplied.
                The $\ell_1$ distance walked from the start point is `depth * sum(trajectory.values())`.
            start: the starting point of the walk operation.
            limit: $L$
            initial_values: the initial values matrix for the limit calculation
            final_projection: the final projection matrix for the limit calculation
        Returns:
            the delta value as defined above.
        """
        depths = list(range(1, depth + 1))
        if limit is None:
            depths += [2 * depth]
            approximants = self.limit(
                trajectory, depths, start, initial_values, final_projection
            )
            limit = approximants[-1].as_float()
            approximants = approximants[:-1]
        else:
            approximants = self.limit(
                trajectory, depths, start, initial_values, final_projection
            )
        return [approximant.delta(limit) for approximant in approximants]
