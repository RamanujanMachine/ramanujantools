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


def test_multiplication():
    precision = 3

    A_coeffs = [Matrix.diag(1, 1), Matrix.diag(2, 2), Matrix.diag(3, 3)]
    B_coeffs = [Matrix.diag(4, 4), Matrix.diag(5, 5), Matrix.diag(6, 6)]

    A = SeriesMatrix(A_coeffs, p=1, precision=precision)
    B = SeriesMatrix(B_coeffs, p=1, precision=precision)

    C = A * B

    # C_0 = A_0 * B_0 = 1 * 4 = 4
    assert C.coeffs[0] == Matrix.diag(4, 4)

    # C_1 = A_0 * B_1 + A_1 * B_0 = 1*5 + 2*4 = 13
    assert C.coeffs[1] == Matrix.diag(13, 13)

    # C_2 = A_0 * B_2 + A_1 * B_1 + A_2 * B_0 = 1*6 + 2*5 + 3*4 = 28
    assert C.coeffs[2] == Matrix.diag(28, 28)


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


def test_shift():
    precision = 10
    matrices = generate_test_matrices()
    dim = matrices[0].shape[0]
    S = SeriesMatrix(matrices, p=1, precision=precision)

    shifted_S = S.shift()

    t = sp.Symbol("t")
    expected_coeffs = [Matrix.zeros(dim) for _ in range(precision)]

    for i, C in enumerate(S.coeffs):
        if C.is_zero_matrix:
            continue

        scalar_series = sp.series((t / (1 + t)) ** i, t, 0, precision).removeO()
        for k in range(precision):
            coeff_val = scalar_series.coeff(t, k)
            if coeff_val != sp.S.Zero:
                expected_coeffs[k] += C * coeff_val

    assert shifted_S.coeffs == expected_coeffs


def test_series_matrix_coboundary():
    precision = 2
    M_coeffs = [Matrix.eye(2), Matrix.zeros(2, 2)]
    M = SeriesMatrix(M_coeffs, p=1, precision=precision)

    Y = Matrix([[0, 1], [0, 0]])
    T_coeffs = [Matrix.eye(2), Y]
    T = SeriesMatrix(T_coeffs, p=1, precision=precision)

    # M is the Identity matrix.
    # M_new = T(n+1)^{-1} I T(n) = T(n+1)^{-1} T(n)
    # T(n+1)^{-1} T(n) evaluates exactly to the Identity matrix + O(t^2)
    M_cob = M.coboundary(T)

    assert M_cob.coeffs[0] == Matrix.eye(2)
    assert M_cob.coeffs[1] == Matrix.zeros(2, 2)


def test_series_matrix_shear_coboundary():
    """
    Validates the analytical discrete shear coboundary S(n+1)^{-1} M(n) S(n).
    Proves that the algebraic shift and the discrete (1+t^p) Taylor correction
    are simultaneously and correctly applied.
    """
    precision = 2
    M_coeffs = [
        Matrix([[1, 0], [0, 1]]),
        Matrix([[0, 1], [1, 0]]),
    ]
    M = SeriesMatrix(M_coeffs, p=1, precision=precision)

    M_sheared, h = M.shear_coboundary(shift=1)

    assert 0 == h
    assert Matrix([[1, 0], [1, 1]]) == M_sheared.coeffs[0]
    assert Matrix([[0, 0], [1, 1]]) == M_sheared.coeffs[1]


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


def test_divide_by_t():
    """
    Tests that divide_by_t correctly shifts the series coefficients left by one
    and pads the tail with a zero matrix to maintain precision.
    """
    dim = 2
    M0 = Matrix([[1, 2], [3, 4]])
    M1 = Matrix([[5, 6], [7, 8]])
    M2 = Matrix([[9, 10], [11, 12]])

    S = SeriesMatrix([M0, M1, M2], p=1, precision=3)

    # We now capture the new matrix!
    S = S.divide_by_t()

    assert S.precision == 3
    assert len(S.coeffs) == 3

    assert S.coeffs[0] == M1
    assert S.coeffs[1] == M2
    assert S.coeffs[2] == Matrix.zeros(dim, dim)


def test_ramify():
    """
    Tests that ramify correctly substitutes t = tau^b, scaling the precision,
    updating the ramification index p, and perfectly spacing out the coefficients.
    """
    dim = 2
    M0 = Matrix([[1, 2], [3, 4]])
    M1 = Matrix([[5, 6], [7, 8]])

    S = SeriesMatrix([M0, M1], p=1, precision=2)

    b = 3
    S_ramified = S.ramify(b)

    assert S_ramified.p == 3
    assert S_ramified.precision == 6
    assert len(S_ramified.coeffs) == 6

    assert S_ramified.coeffs[0] == M0  # tau^0
    assert S_ramified.coeffs[1] == Matrix.zeros(dim)
    assert S_ramified.coeffs[2] == Matrix.zeros(dim)
    assert S_ramified.coeffs[3] == M1  # tau^3
    assert S_ramified.coeffs[4] == Matrix.zeros(dim)
    assert S_ramified.coeffs[5] == Matrix.zeros(dim)
