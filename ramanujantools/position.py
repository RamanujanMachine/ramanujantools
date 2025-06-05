from __future__ import annotations

import sympy as sp


class Position(dict[sp.Symbol, sp.Expr]):
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

    def __iadd__(self, other: dict):
        for key in other:
            self[key] = self.get(key, 0) + other[key]
        return self

    def __add__(self, other: dict):
        result = self.copy()
        result += other
        return result

    def __isub__(self, other: dict):
        self += -other
        return self

    def __sub__(self, other: dict):
        result = self.copy()
        result -= other
        return result

    def __mul__(self, coefficient: int):
        return Position({key: coefficient * value for key, value in self.items()})

    def __rmul__(self, coefficient: int):
        return self * coefficient

    def __neg__(self):
        return -1 * self

    def __hash__(self):
        return hash(frozenset(self.items()))

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

    def is_polynomial(self) -> bool:
        """
        Returns true iff all position elements are polynomial
        """
        for element in self.values():
            if not all(
                [
                    c.is_Integer
                    for c in sp.simplify(element).as_coefficients_dict().values()
                ]
            ):
                return False
        return True

    def denominator_lcm(self) -> int:
        r"""
        Returns the lcm of all denominators
        """
        return int(
            sp.lcm(
                [sp.simplify(element).as_numer_denom()[1] for element in self.values()]
            )
        )

    def is_integer(self) -> bool:
        """
        Returns true iff all position elements are numerical
        """
        return all(sp.simplify(element).is_Integer for element in self.values())

    def free_symbols(self) -> set:
        symbols = set()
        for value in self.values():
            symbols = symbols.union((sp.S(0) + value).free_symbols)
        return symbols

    def sorted(self) -> Position:
        return Position({key: self[key] for key in sorted(self.keys(), key=str)})

    @staticmethod
    def from_list(values: list[sp.Expr], symbol: str) -> Position:
        """
        Constructs a Position object with incrementing symbols from a list.
        Example:
            >>> Position.from_list([1, 2, 3], "x")
            Position({x0: 1, x1: 2, x2: 3})
        """
        amount = len(values)
        symbols = sp.symbols(f"{symbol}:{amount}")
        return Position({symbols[i]: values[i] for i in range(amount)})
