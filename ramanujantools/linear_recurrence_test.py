from sympy.abc import n
import sympy as sp

from ramanujantools import LinearRecurrence, Matrix
from ramanujantools.pcf import PCF


def f(c, index=n):
    return sp.Function(c)(n).subs({n: index})


def test_repr():
    expected = "LinearRecurrence([n, 1, 3 - n**2])"
    r = eval(expected)
    assert expected == repr(r)


def test_relation():
    expected = [1, n, n**2, n**3 - 7, 13 * n - 12]
    assert expected == LinearRecurrence(expected).relation


def test_matrix():
    relation = [1, n, n + 1]
    assert Matrix([[0, n + 1], [1, n]]) == LinearRecurrence(relation).recurrence_matrix


def test_limit():
    initial_values = Matrix([[2, 3, 5], [7, 11, 13]])
    r = LinearRecurrence([1, n + 1, n**2])
    depths = [2, 3, 19, 101]
    assert r.limit(depths, 0, initial_values) == r.recurrence_matrix.limit(
        {n: 1}, depths, {n: 0}, initial_values
    )


def test_evaluate_solution_fibonacci():
    r = LinearRecurrence([1, 1, 1])
    given_index = 0  # i.e, the initial values are at indices -1 and 0.
    indices = list(range(given_index + 1, given_index + 6))
    initial_values = Matrix([[1, 1]])
    assert r.evaluate_solution(initial_values, indices, given_index) == [2, 3, 5, 8, 13]


def test_evaluate_solution_basis_single():
    r = LinearRecurrence([1, n, n**2, 3])
    indices = 100
    initial_values = Matrix([[0, 1, 0]])
    matrix = r.recurrence_matrix.walk({n: 1}, indices, {n: 0})
    evaluation = r.evaluate_solution(initial_values, indices)
    assert initial_values.dot(matrix.col(-1)) == evaluation


def test_evaluate_solution_basis_list_with_given_index():
    r = LinearRecurrence([n, -n, 5, n**2 + n + 1])
    given_index = 3
    max_index = 100
    initial_values = Matrix([[0, 0, 1]])
    indices = list(range(given_index + 1, 100))
    matrices = r.recurrence_matrix.walk(
        {n: 1}, list(range(1, max_index - given_index)), {n: given_index}
    )
    evaluations = r.evaluate_solution(initial_values, indices, given_index)
    assert len(matrices) == len(evaluations)
    for i in range(len(evaluations)):
        print(i)
        assert initial_values.dot(matrices[i].col(-1)) == evaluations[i]


def test_evaluate_solution_generic():
    r = LinearRecurrence([2, -n + 1, n**3 + n, 14])
    given_index = 17
    max_index = 123
    initial_values = Matrix([[5, -4, 2]])
    indices = list(range(given_index + 1, max_index))
    matrices = r.recurrence_matrix.walk(
        {n: 1}, list(range(1, max_index - given_index)), {n: given_index}
    )
    evaluations = r.evaluate_solution(initial_values, indices, given_index)
    assert len(matrices) == len(evaluations)
    for i in range(len(evaluations)):
        assert initial_values.dot(matrices[i].col(-1)) == evaluations[i]


def test_inflate():
    r = LinearRecurrence([f("a"), f("b"), f("c"), f("d")])
    assert LinearRecurrence(
        [
            f("a"),
            f("b") * f("e"),
            f("c") * f("e") * f("e", n - 1),
            f("d") * f("e") * f("e", n - 1) * f("e", n - 2),
        ]
    ) == r.inflate(f("e"))


def test_deflate():
    r = LinearRecurrence([f("a"), f("b"), f("c"), f("d")])
    assert r.inflate(1 / f("e")) == r.deflate(f("e"))


def test_fold():
    r = LinearRecurrence([f("a"), f("b"), f("c")])
    assert LinearRecurrence(
        [
            f("a"),
            -f("a", n - 1) * f("d") + f("b"),
            f("b", n - 1) * f("d") + f("c"),
            f("c", n - 1) * f("d"),
        ]
    ) == r.fold(f("d"))


def test_unfold_poly():
    r = LinearRecurrence([1, n, n])
    multiplier = n - 3
    folded = r.fold(multiplier)
    assert r == folded.unfold_poly(multiplier)


def test_unfold():
    r = LinearRecurrence([1, n, (n - 3) * (n + 7) * (n - 11)])
    multiplier = (n - 1) * (n + 2)
    folded = r.fold(multiplier)
    assert (r, sp.Poly(multiplier, n)) == folded.unfold()[0]


def test_unfold_deflate():
    r = LinearRecurrence([1, n, n])
    multiplier = n - 17
    inflation = n + 1
    folded = r.fold(multiplier).inflate(inflation)
    solutions = folded.unfold(-1)
    recurrence, actual_multiplier = solutions[-1]
    assert r == recurrence.deflate(inflation)
    assert actual_multiplier == sp.Poly(multiplier * inflation, n)


def test_gamma():
    # from https://arxiv.org/abs/1010.1420
    m = Matrix([[0, -(n**2), 0], [1, 2 * n + 2, 0], [0, -(n - 1) / (n + 1), 1]])
    r = LinearRecurrence(m)
    assert [] == r.unfold()
    lim = r.limit(1000, 2)
    assert Matrix([[3, 12, 59], [6, 21, 102]]) == lim.identify(lim.mp.euler)
    assert Matrix([[1, 4, 20], [2, 7, 34]]) == lim.identify(
        -lim.mp.e * lim.mp.ei(-1)  # Gompertz
    )
    inflated = r.inflate(2 - n)

    lim = inflated.limit(1000, 5)
    assert Matrix([[3625, -26792, 461718], [6270, -46380, 799620]]) == lim.identify(
        lim.mp.euler, maxcoeff=10**6
    )

    unfolded = inflated.unfold_poly(n)
    pcf = PCF(unfolded.recurrence_matrix).deflate_all()
    assert PCF(-2 * (n + 2), -((n + 1) ** 2)) == pcf


def test_apteshuffle():
    expected = LinearRecurrence(  # from the paper
        [
            (16 * n - 15) * (16 * n + 1),
            (16 * n - 15) * (256 * n**3 + 528 * n**2 + 352 * n + 73),
            -(16 * n + 17) * (128 * n**3 + 40 * n**2 - 82 * n - 45),
            n**2 * (16 * n + 1) * (16 * n + 17),
        ]
    )

    apt = LinearRecurrence(  # from the paper
        [
            16 * n - 15,
            128 * n**3 + 40 * n**2 - 82 * n - 45,
            -(n**2) * (256 * n**3 - 240 * n**2 + 64 * n - 7),
            n**2 * (n - 1) ** 2 * (16 * n + 1),
        ]
    )

    assert expected == apt.apteshuffle().deflate((n + 1) ** 2)
