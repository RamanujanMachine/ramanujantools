from sympy.abc import x, y

from matrix import Matrix
from cmf import CMF

e = CMF(Matrix([[1, -y - 1], [-1, x + y + 2]]), Matrix([[0, -y - 1], [-1, x + y + 1]]))

pi = CMF(
    Matrix([[x, -x], [-y, 2 * x + y + 1]]),
    Matrix([[1 + y, -x], [-1 - y, x + 2 * y + 1]]),
)

zeta3 = CMF(
    Matrix(
        [
            [0, -(x**3)],
            [(x + 1) ** 3, x**3 + (x + 1) ** 3 + 2 * y * (y - 1) * (2 * x + 1)],
        ]
    ),
    Matrix(
        [
            [-(x**3) + 2 * x**2 * y - 2 * x * y**2 + y**3, -(x**3)],
            [x**3, x**3 + 2 * x**2 * y + 2 * x * y**2 + y**3],
        ]
    ),
)
