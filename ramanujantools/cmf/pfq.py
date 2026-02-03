from __future__ import annotations

import sympy as sp
from sympy.abc import z

from ramanujantools import Matrix, Position
from ramanujantools.cmf import DFinite
from ramanujantools.cmf.d_finite import theta


class pFq(DFinite):
    r"""
    Represents the pFq CMF, derived from the differentiation property of generalized hypergeometric functions:
    https://en.wikipedia.org/wiki/Generalized_hypergeometric_function
    """

    def __init__(
        self,
        p: int,
        q: int,
        z: sp.Expr = z,
    ):
        r"""
        Constructs a pFq CMF.
        Args:
            p: The number of numerator parameters in the hypergeometric function
            q: The number of denominator parameters in the hypergeometric function
            z: If given, will attempt to construct the CMF for a specific z value.
        """
        self.p = p
        self.q = q
        self.z = sp.S(z)
        super().__init__(p, q, z)

    def __repr__(self) -> str:
        return f"pFq({self.p, self.q, self.z})"

    @staticmethod
    def x_axes(p: int) -> list[sp.Symbol]:
        return sp.symbols(f"x:{p}")

    @staticmethod
    def y_axes(q: int) -> list[sp.Symbol]:
        return sp.symbols(f"y:{q}")

    @classmethod
    def axes_and_signs(cls, p, q, z) -> dict[sp.Expr, bool]:
        x_axes = {x_i: True for x_i in pFq.x_axes(p)}
        y_axes = {y_i: False for y_i in pFq.y_axes(q)}
        return {**x_axes, **y_axes}

    @classmethod
    def differential_equation(cls, p, q, z) -> sp.Poly:
        return sp.Poly(
            sp.expand(
                theta * sp.prod(theta + y_i - 1 for y_i in pFq.y_axes(q))
                - z * sp.prod(theta + x_i for x_i in pFq.x_axes(p))
            ),
            theta,
        )

    @classmethod
    def construct_matrix(
        cls, theta_matrix: Matrix, axis: sp.Symbol, *args, **kwargs
    ) -> Matrix:
        eye = Matrix.eye(theta_matrix.rows)
        denom = axis - 1 if axis.name.startswith("y") else axis
        return theta_matrix / denom + eye

    def ascend(
        self, trajectory: Position, start: Position
    ) -> tuple[pFq, Position, Position]:
        r"""
        Returns a tuple of (CMF, trajectory, start), such that:
        1. The CMF is ascended, i.e, the CMF of _{p+1}F_{q+1}
        2. The start position contains two new more symbols
        3. The trajectory is padded with two zeros
        such that for any two choices of parameters $x_p, y_p$ such that $x_p - y_p \in \mathbb{Z}$,
        the ascended trajectory matrix contains all constants of the original,
        and the ascended delta is the same as the original (in a type-2 limit context).
        """
        ascended_cmf = pFq(
            self.p + 1,
            self.q + 1,
            self.z,
        )
        xp = sp.Symbol(f"x{self.p}")
        yq = sp.Symbol(f"y{self.q}")
        ascended_start = start + Position({xp: xp, yq: yq})
        ascended_trajectory = trajectory + Position({xp: 0, yq: 0})
        return (ascended_cmf, ascended_trajectory.sorted(), ascended_start.sorted())

    def subs(self, substitutions: Position) -> pFq:
        self._validate_axes_substitutions(substitutions)
        return pFq(
            self.p,
            self.q,
            self.z.subs(substitutions),
        )

    @staticmethod
    def predict_rank(p: int, q: int, z: sp.Expr):
        """
        Returns the rank of the CMF (i.e, the rank of its matrices).
        """
        N = max(p, q + 1)
        if z == 1 and p == q + 1:
            N -= 1
        return N

    @staticmethod
    def theta_derivative(expr: sp.Expr):
        r"""
        Returns the \theta derivative of an expression,
        which is defined as z * d/dz.
        """
        return z * sp.Derivative(expr, z).simplify()

    @staticmethod
    def state_vector(
        a_values: list[sp.Rational], b_values: list[sp.Rational], z_eval: sp.Expr = z
    ):
        r"""
        Returns the state vector of a pFq CMF in a specific point.
        The state vector is of length N, and the ith element is $\theta^i pFq(\bar{a}, \bar{b}, z)$,
        with a_values, b_values and z substituted in.
        """
        p = len(a_values)
        q = len(b_values)
        a_symbols = sp.symbols(f"a:{p}")
        b_symbols = sp.symbols(f"b:{q}")
        values = [sp.hyper(a_symbols, b_symbols, z)]
        for _ in range(1, pFq.predict_rank(p, q, z_eval)):
            values.append(pFq.theta_derivative(values[-1]))
        a_subs = Position.from_list(a_values, "a")
        b_subs = Position.from_list(b_values, "b")
        return sp.hyperexpand(
            Matrix(values).transpose().subs(a_subs | b_subs | {z: z_eval})
        )

    @staticmethod
    def contiguous_relation(point, anchor, z) -> Matrix:
        a_point, b_point = point
        a_anchor, b_anchor = anchor
        p = len(a_point)
        q = len(b_point)
        start = Position.from_list(a_anchor, "x") | Position.from_list(b_anchor, "y")
        end = Position.from_list(a_point, "x") | Position.from_list(b_point, "y")
        return pFq(p, q, z).work(start, end)

    @staticmethod
    def evaluate(
        a_values: list[sp.Rational], b_values: list[sp.Rational], z_eval: sp.Rational
    ) -> sp.Expr:
        """
        Evaluates symbolically the pFq function at a specific point.
        Levarages the pFq CMF to calculate the contiguous relations.
        Works by selecting a small anchor point for which we calculate the state vector
        using sympy, and then calculating the `work` matrix which represents the contiguous
        relations transformation up to our desired point.
        """
        a_values = [sp.S(value) for value in a_values]
        b_values = [sp.S(value) for value in b_values]
        a_anchor = [
            sp.sign(value) * (value - (value.floor() - 1)) for value in a_values
        ]
        b_anchor = [
            sp.sign(value) * (value - (value.floor() - 2)) for value in b_values
        ]
        vector = pFq.state_vector(a_anchor, b_anchor, z_eval)
        m = pFq.contiguous_relation((a_values, b_values), (a_anchor, b_anchor), z_eval)
        return (vector * m)[0]

    def determinant(self, axis: sp.Symbol):
        """
        Returns the determinant of an axis (basis) matrix in factored form via
        a hardcoded formula, for quick performance.

        We have the following differential equation for the generalized hypergeometric function
        $pFq(x_0,...x_{p-1},y_0,\\ldots y_{q-1}|z)$:

        $$\\theta_z \\prod_{i=0}^{q-1}(\\theta_z+y_i-1)-z\\prod_{i=0}^{p-1}(\\theta_z+x_i)=0$$

        We have the following contiguous relations:
        $$\\theta_z = x_i S_{x_i} + x_i, i=0,\\ldots,p-1$$
        $$\\theta_z = (y_i-1) S_{y_i} + y_i-1, i=0,\\ldots,q-1$$
        For any shift $S\\in \\{S_{x_0},\\ldots,S_{x_{p-1}}, S_{y_0},\\ldots,S_{y_{q-1}}\\}$,
        we substitute the contiguous relation in the differential equation and obtain a polynomial in $S$.
        We expect the obtained polynomial in $S$ to be a characteristic polynomial for the operator $S$
        (after dividing by the lead coefficient).
        The free coefficient of the normalized polynomial should be $det([S])$ times (-1) taken to
        the dimension of the matrix.

        x axis calculations:
        For $p=q+1$ we obtain:
        $$(-1)^{p-1}\\frac{\\prod_{i=0}^{q-1}(-x_r+y_i-1)}{(1-z)x_r^{p-1}}$$

        For $p>q+1$ the determinant will be:
        $$(-1)^p\\frac{\\prod_{i=0}^{q-1}(-x_r+y_i-1)}{zx_r^{p-1}}.$$

        For $p < q+1$ the determinant will be:
        $$(-1)^{q}\\frac{\\prod_{i=0}^{q-1}(-x_r+y_i-1)}{x_r^{q}}.$$
        (power is $q$ because denominator is $-x_r^{q}$)

        y axis parameters: We calculate as before but in order to get the determinant for
        $S_r$ from $S_r^{-1}$ we invert and replace $y_0$ with $y_0+1$:

        For $p=q+1$ we obtain:

        $$(-1)^p\\frac{(1-z)(y_r)^{p}}{-z\\prod_{i=0}^{p-1}(-y_r+x_i)}$$

        For $p>q+1$:
        $$(-1)^p\\frac{(y_r)^p}{\\prod_{i=0}^{p-1}(-y_r+x_i)}.$$

        For $p < q+1$:
        $$(-1)^{q+1}\\frac{y_r^{q+1}}{-z\\cdot\\prod_{i=0}^{p-1}(-y_r+x_i)}$$

        p=q+1 and z=1 case: For $p=q+1$ and $z=1$, $S_r$ is a zero of the following expression:
        $$(x_r S_r - x_r)\\prod_{i=0}^{q-1}((x_r S_r - x_r)+y_i-1)-\\prod_{i=0}^{q}(x_r S_r - x_r+x_i)=0$$

        We obtain that the determinant is:
        $$(-1)^q\\frac{-x_r\\prod_{i=0}^{q-1}(-x_r+y_i-1)}{ x_r^{q+1} -
        x_r^{q}(\\sum_{i=0}^{q-1}(y_i)-\\sum_{i=0}^q(x_i)+x_r-q)}$$

        For the $y_i$ shift, we substitute $\\theta = (y_r-1)S-(y_r-1)$.
        $$((y_r-1) S_r^{-1} -y_r+1)\\prod_{i=0}^{q-1}((y_r-1) S_r^{-1} -y_r+y_i) -
        \\prod_{i=0}^{p-1}((y_r-1) S_r^{-1} -y_r+1+x_i)=0$$

        and we obtain for the determinant::
        $$(-1)^{q+1} \\frac{-(y_r)^{q+1}+(y_r)^{q-1}((\\sum_{i=0}^{q-1}y_i + \\sum x_i ) +y_r-q+1)}
        {\\prod_{i=0}^{p-1}(-y_r+x_i)}$$
        """
        p, q, z = self.p, self.q, self.z
        is_y = axis.name.startswith("y")
        x_axes = pFq.x_axes(p)
        y_axes = pFq.y_axes(q)

        prod_x = sp.prod(-axis + x_i for x_i in x_axes)
        prod_y = sp.prod(-axis + y_i - 1 for y_i in y_axes)

        if p == q + 1 and z == 1:
            sum_diff = sum(y_axes) - sum(x_axes) - q + axis

            if is_y:
                poly = -(axis ** (q + 1)) + axis**q * (sum_diff + 1)
                return sp.factor(((-1) ** (q + 1) * poly) / prod_x)
            else:
                poly = axis ** (q + 1) - axis**q * sum_diff
                return sp.factor(((-1) ** (q + 1) * -axis * prod_y) / poly)

        if p == q + 1:
            if is_y:
                term = ((-1) ** p * (1 - z) * axis**p) / -z
            else:
                term = ((-1) ** (p - 1)) / ((1 - z) * axis ** (p - 1))

        elif p > q + 1:
            if is_y:
                term = (-1) ** p * axis**p
            else:
                term = ((-1) ** p) / (z * axis ** (p - 1))

        else:
            if is_y:
                term = ((-1) ** (q + 1) * axis ** (q + 1)) / -z
            else:
                term = ((-1) ** q) / axis**q

        if is_y:
            return sp.factor(term / prod_x)
        else:
            return sp.factor(term * prod_y)
