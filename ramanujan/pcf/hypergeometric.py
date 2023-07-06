import sympy as sp
from sympy.abc import n

from ramanujan.pcf import PCF


class HypergeometricLimit:
    """Factory class that creates an instance of a hpyergeometric limit class (either 1f1 or 2f1)."""

    @staticmethod
    def __call__(pcf: PCF):
        supported_degrees = {
            (1, 1): Hypergeometric1F1Limit,
            (1, 2): Hypergeometric2F1Limit,
        }
        if pcf.degree() not in supported_degrees:
            raise ValueError(
                (
                    "Attempted to evaluate hypergeometric limit of {} of degree {}. "
                    "Supported degrees are {}"
                ).format(pcf, pcf.degree(), supported_degrees.keys())
            )
        return supported_degrees[pcf.degree](pcf)


class HypLimitInterface:
    def __init__(self, pcf: PCF):
        pass

    def limit(self):
        """Attempts to calculate the limit using sympy's hypergeometric functions"""
        pass

    def as_mathematica_prompt(self):
        """Returns a mathematica prompt that will calculate the limit of this pcf"""
        pass


class Hypergeometric1F1Limit(HypLimitInterface):
    def __init__(self, pcf: PCF):
        [a1, a0] = sp.Poly(pcf.a_n, n).all_coeffs()
        [b1, b0] = sp.Poly(pcf.b_n, n).all_coeffs()

        self.alpha = b0 / b1
        self.beta = (a0 * a1 + b1) / a1**2
        self.z = b1 / a1**2

    def limit(self):
        """Attempts to calculate the limit using sympy's 1f1 functions"""
        return sp.simplify(
            sp.hyperexpand(
                self.alpha
                * self.beta
                * (
                    sp.hyper([self.alpha], [self.beta], self.z)
                    / sp.hyper([self.alpha + 1], [self.beta + 1], self.z)
                )
            )
        )

    def as_mathematica_prompt(self):
        """Returns a mathematica prompt that will calculate the limit of this pcf"""
        return (
            f"{self.alpha} * {self.beta} * "
            f"Hypergeometric1F1[{self.alpha}, {self.beta}, {self.z}] / "
            f"Hypergeometric1F1[{self.alpha + 1}, {self.beta + 1}, {self.z + 1}]"
        )


class Hypergeometric2F1Limit(HypLimitInterface):
    def __init__(self, pcf: PCF):
        [e, d] = sp.Poly(pcf.a_n, n).all_coeffs()
        [c, b, a] = sp.Poly(pcf.b_n, n).all_coeffs()
        delta = e**2 + 4 * c
        # assert delta > 0, "Delta is less than 0!"
        self.sqrt_delta = sp.sign(e) * sp.root(delta, 2)
        self.alpha, self.beta = list(sp.solve(c * n**2 - b * n + a, n))
        self.z = sp.simplify((1 - e / self.sqrt_delta) / 2)
        self.gamma = sp.simplify(((b + c) * self.z) / c + d / self.sqrt_delta)

    def limit(self):
        """Attempts to calculate the limit using sympy's 2f1 functions"""
        return sp.simplify(
            sp.hyperexpand(
                self.sqrt_delta
                * self.gamma
                * (
                    sp.hyper([self.alpha, self.beta], [self.gamma], self.z)
                    / sp.hyper(
                        [self.alpha + 1, self.beta + 1], [self.gamma + 1], self.z
                    )
                )
            )
        )

    def as_mathematica_prompt(self):
        """Returns a mathematica prompt that will calculate the limit of this pcf"""
        return (
            f"{self.sqrt_delta} * {self.gamma} * "
            f"Hypergeometric2F1[{self.alpha}, {self.beta}, {self.gamma}, {self.z}] / "
            f"Hypergeometric2F1[{self.alpha + 1}, {self.beta + 1}, {self.gamma + 1}, {self.z}]"
        )
