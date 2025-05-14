import pytest
import sympy
from sympy.abc import x, y, z
from ramanujantools.generic_polynomial import GenericPolynomial


@pytest.mark.parametrize(
    "n, num_var, output",
    [
        (5, 1, [[5]]),
        (5, 2, [[0, 5], [1, 4], [2, 3], [3, 2], [4, 1], [5, 0]]),
        (
            5,
            3,
            [
                [0, 0, 5],
                [0, 1, 4],
                [0, 2, 3],
                [0, 3, 2],
                [0, 4, 1],
                [0, 5, 0],
                [1, 0, 4],
                [1, 1, 3],
                [1, 2, 2],
                [1, 3, 1],
                [1, 4, 0],
                [2, 0, 3],
                [2, 1, 2],
                [2, 2, 1],
                [2, 3, 0],
                [3, 0, 2],
                [3, 1, 1],
                [3, 2, 0],
                [4, 0, 1],
                [4, 1, 0],
                [5, 0, 0],
            ],
        ),
    ],
)
def test_sum_to(n: int, num_var: int, output: list[list[int]]):
    for elements in GenericPolynomial._sum_to(n=n, num_var=num_var):
        assert elements in output
        output.remove(elements)

    assert len(output) == 0


def test_generic_of_degree():
    f, _ = GenericPolynomial.of_degree(deg=3, var_name="f_", s=x, monic=False)
    f_0, f_1, f_2, f_3 = sympy.symbols("f_0 f_1 f_2 f_3")
    assert f == f_0 + f_1 * x + f_2 * x**2 + f_3 * x**3

    f, _ = GenericPolynomial.of_degree(deg=3, var_name="f_", s=x, monic=True)
    assert f == f_0 + f_1 * x + f_2 * x**2 + x**3


def test_generic_of_combined_degree():
    f, _ = GenericPolynomial.of_combined_degree(deg=2, var_name="f_", variables=[x, y])
    # sympy.symbols consider comma (,) as a separator, so need to use sympy.Symbol instead
    f_00 = sympy.Symbol("f_(0,0)")
    f_10 = sympy.Symbol("f_(1,0)")
    f_01 = sympy.Symbol("f_(0,1)")
    f_20 = sympy.Symbol("f_(2,0)")
    f_11 = sympy.Symbol("f_(1,1)")
    f_02 = sympy.Symbol("f_(0,2)")
    assert f == f_00 + f_10 * x + f_01 * y + f_20 * x**2 + f_11 * x * y + f_02 * y**2


def test_symmetric_poly():
    symm0, symm1, symm2, symm3 = GenericPolynomial.symmetric_polynomials(x, y, z)
    assert symm0 == 1
    assert symm1 == x + y + z
    assert symm2 == x * y + y * z + z * x
    assert symm3 == x * y * z


def test_as_symmetric():
    x, y, z, w = sympy.symbols("x y z w")
    s1, s2, s3, s4 = sympy.symbols("s_1 s_2 s_3 s_4")

    # two symmetric variables
    p = (x + y) * (x * y) * z + z**2 * (x**2 + y**2) + (x + y) + 1
    symm_p = s1 * s2 * z + z**2 * (s1**2 - 2 * s2) + s1 + 1
    result = GenericPolynomial.as_symmetric(
        polynomial=p, symm_symbols=[x, y], symm_var_name="s_"
    )
    assert sympy.simplify(result - symm_p) == 0

    # three symmetric variables
    p = (x + y + z) * (x * y * z) * w + w**2 * (x**2 + y**2 + z**2) + (x + y + z) + 1
    symm_p = s1 * s3 * w + w**2 * (s1**2 - 2 * s2) + s1 + 1
    result = GenericPolynomial.as_symmetric(
        polynomial=p, symm_symbols=[x, y, z], symm_var_name="s_"
    )
    assert sympy.simplify(result - symm_p) == 0
