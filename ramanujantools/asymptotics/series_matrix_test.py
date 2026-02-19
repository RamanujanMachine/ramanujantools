import sympy as sp
import random

from ramanujantools import Matrix
from ramanujantools.asymptotics import SeriesMatrix


def generate_test_matrices(seed=42, dim=4, count=8):
    random.seed(seed)
    matrices = []
    for i in range(count):
        mat = Matrix(dim, dim, lambda r, c: random.randint(-5, 5))
        if i == 0:
            mat = mat + 20 * Matrix.eye(dim)
        matrices.append(mat)
    return matrices


def test_construction():
    eye = Matrix.eye(2)
    p = 1
    precision = 4
    A = Matrix([[1, 2], [3, 4]])
    S = SeriesMatrix([eye, A], p=p, precision=precision)
    assert eye == S.coeffs[0]
    assert A == S.coeffs[1]
    assert p == S.p
    assert precision == S.precision
    for i in range(2, precision):
        assert Matrix.zeros(2, 2) == S.coeffs[i]


def test_inverse(precision=10):
    matrices = generate_test_matrices()
    dim = matrices[0].shape[0]
    S = SeriesMatrix(matrices, p=1, precision=precision)

    V = S.inverse()
    Identity_Check = S * V

    for i in range(precision):
        expected = Matrix.eye(dim) if i == 0 else Matrix.zeros(dim)
        assert Identity_Check.coeffs[i] == expected, (
            f"Inverse mismatch at coefficient t^{i}"
        )


def test_shift(precision=10):
    matrices = generate_test_matrices()
    dim = matrices[0].shape[0]
    S = SeriesMatrix(matrices, p=1, precision=precision)

    shifted_S = S.shift()

    t = sp.Symbol("t")
    t_new = sp.series(t / (1 + t), t, 0, precision).removeO()

    expected_sym = Matrix.zeros(dim)
    for i, C in enumerate(matrices):
        expected_sym += C * (t_new**i)

    expected_sym = expected_sym.applyfunc(
        lambda x: sp.expand(x).series(t, 0, precision).removeO()
    )

    algorithmic_sym = Matrix.zeros(dim)
    for i, C in enumerate(shifted_S.coeffs):
        algorithmic_sym += C * (t**i)

    diff = (expected_sym - algorithmic_sym).factor()
    assert diff == Matrix.zeros(dim), (
        "Shift output does not match SymPy analytic Taylor expansion."
    )


def test_series_matrix_valuations():
    """
    Tests that SeriesMatrix correctly identifies the lowest non-zero
    power (valuation) for every cell in the matrix series.
    """
    # M_0 has a value only at (0, 0)
    C0 = Matrix([[1, 0], [0, 0]])

    # M_1 has a value only at (0, 1)
    C1 = Matrix([[0, 2], [0, 0]])

    # M_2 has a value only at (1, 0)
    C2 = Matrix([[0, 0], [3, 0]])

    # Cell (1, 1) remains 0 across all coefficients

    # Construct the series: C0 + C1*t + C2*t^2
    SM = SeriesMatrix([C0, C1, C2], precision=3)

    vals = SM.valuations()

    # Assertions
    assert vals[0, 0] == 0  # Appeared in C0
    assert vals[0, 1] == 1  # Appeared in C1
    assert vals[1, 0] == 2  # Appeared in C2
    assert vals[1, 1] == sp.oo  # Never appeared

    print("Valuations correctly extracted!")
