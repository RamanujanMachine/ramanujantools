from __future__ import annotations
from typing import Dict

import sympy as sp


class Position(dict):
    r"""
    Represents a Position (or a trajectory).

    Inherits from dict and adds algebraic utilities.

    Example:
        >>> p = Position({x: 1, y: 2, z: 3})
        >>> p
        Position({x: 1, y: 2, z: 3})
        >>> 2 * p
        Position({x: 2, y: 4, z: 6})
        >>> trajectory = Position({x: 3, y: -1, z: 2})
        >>> p += 4 * trajectory
        Position({x: 13, y: -2, z: 11})
    """

    def __repr__(self):
        return f"Position({super().__repr__()})"

    def __iadd__(self, other: Dict):
        for key in other:
            self[key] = self.get(key, 0) + other[key]
        return self

    def __add__(self, other: Dict):
        result = self.copy()
        result += other
        return result

    def __isub__(self, other: Dict):
        self += -other
        return self

    def __sub__(self, other: Dict):
        result = self.copy()
        result -= other
        return result

    def __mul__(self, coefficient: int):
        return Position({key: coefficient * value for key, value in self.items()})

    def __rmul__(self, coefficient: int):
        return self * coefficient

    def __neg__(self):
        return -1 * self

    def copy(self):
        return Position(super().copy())

    def longest(self):
        r"""
        Returns the longest element of `this`, i.e, the largest (abs) value in it.
        This is the $ L_\infty $ norm of the vector.
        """
        return max([abs(value) for value in self.values()], default=0)

    def shortest(self):
        r"""
        Returns the shortest nonzero element of `this`, i.e, the smallest (abs) nonzero value in it.
        """
        return min([abs(value) for value in self.values() if value != 0], default=0)

    def signs(self):
        r"""
        Returns the sign in each direction of this
        """
        return Position({key: int(sp.sign(value)) for key, value in self.items()})
