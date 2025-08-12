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
    relation = [-1, n, n + 1]
    assert Matrix([[0, n + 1], [1, n]]) == LinearRecurrence(relation).recurrence_matrix


def test_limit():
    initial_values = Matrix([[2, 3, 5], [7, 11, 13]])
    r = LinearRecurrence([1, n + 1, n**2])
    depths = [2, 3, 19, 101]
    assert r.limit(depths, 0, initial_values) == r.recurrence_matrix.limit(
        {n: 1}, depths, {n: 0}, initial_values
    )


def test_evaluate_solution_fibonacci():
    r = LinearRecurrence([-1, 1, 1])
    start = 33
    end = start + 5
    initial_values = Matrix([[1, 1]])
    assert r.evaluate_solution(initial_values, start, end) == [2, 3, 5, 8, 13]


def test_evaluate_solution_basis_single():
    r = LinearRecurrence([1, n, n**2, 3])
    start = 0
    end = 100
    initial_values = Matrix([[0, 1, 0]])
    matrix = r.recurrence_matrix.walk({n: 1}, end, {n: start})
    evaluation = r.evaluate_solution(initial_values, start, end)
    assert initial_values.dot(matrix.col(-1)) == evaluation[-1]


def test_evaluate_solution_basis_list_with_given_index():
    r = LinearRecurrence([n, -n, 5, n**2 + n + 1])
    start = 3
    end = 100
    initial_values = Matrix([[0, 0, 1]])
    matrices = r.recurrence_matrix.walk(
        {n: 1}, list(range(1, end - start + 1)), {n: start}
    )
    evaluations = r.evaluate_solution(initial_values, start, end)
    assert len(matrices) == len(evaluations)
    for i in range(len(evaluations)):
        assert initial_values.dot(matrices[i].col(-1)) == evaluations[i]


def test_evaluate_solution_generic():
    r = LinearRecurrence([2, -n + 1, n**3 + n, 14])
    start = 17
    end = 123
    initial_values = Matrix([[5, -4, 2]])
    matrices = r.recurrence_matrix.walk(
        {n: 1}, list(range(1, end - start + 1)), {n: start}
    )
    evaluations = r.evaluate_solution(initial_values, start, end)
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
            f("a", n - 1) * f("d") + f("b"),
            f("b", n - 1) * f("d") + f("c"),
            f("c", n - 1) * f("d"),
        ]
    ) == r.fold(f("d"))


def test_unfold_poly():
    r = LinearRecurrence([-1, n, n])
    multiplier = n - 3
    folded = r.fold(multiplier)
    assert [r] == folded.unfold_poly(multiplier)


def test_unfold():
    r = LinearRecurrence([-1, n, (n - 3) * (n + 7) * (n - 11)])
    multiplier = (n - 1) * (n + 2)
    folded = r.fold(multiplier)
    assert (r, sp.Poly(multiplier, n)) == folded.unfold()[0]


def test_fold_solution_space():
    r = LinearRecurrence([n, n + 1, (n + 17) ** 2, 3])
    initial_values = Matrix([[1, 2, 3]])
    start = 5
    end = 10
    solution = r.evaluate_solution(initial_values, start, end)
    multiplier = n**2 + n + 1
    folded = r.fold(multiplier)
    assert folded == r + multiplier * r._shift(1)
    folded_initial_values = Matrix.hstack(initial_values, Matrix([solution[0]]))
    shifted_solution = folded.evaluate_solution(folded_initial_values, start + 1, end)
    assert solution[1:] == shifted_solution


def test_unfold_deflate():
    r = LinearRecurrence([-1, n, n])
    multiplier = n - 17
    inflation = n + 1
    folded = r.fold(multiplier).inflate(inflation)
    solutions = folded.unfold(-1)
    recurrence, actual_multiplier = solutions[-1]
    assert r == recurrence.deflate(inflation)
    assert actual_multiplier == sp.Poly(multiplier * inflation, n)


def test_compose_solution_space_constants():
    r = LinearRecurrence([-1, 1, 1])
    initial_values = Matrix([[1, 1]])
    start = 0
    end = 5
    assert [2, 3, 5, 8, 13] == r.evaluate_solution(initial_values, start, end)

    rr = r.compose(r)
    new_initial_values = Matrix([[1, 1, 2, 3]])
    assert [5, 8, 13, 21, 34] == rr.evaluate_solution(new_initial_values, start, end)


def test_compose_solution_space_polynomials():
    r1 = LinearRecurrence([n**2 + 3 * n, 5 * n - 7, 13 * n**3, 2])
    r2 = LinearRecurrence([17 * n, 18 * (n - 2), 19 * (n - 3), 20 * (n - 5)])
    initial_values = Matrix([[17, 18, 19]])
    start = 17
    end = 100
    solution = r2.evaluate_solution(initial_values, start, end)

    rr = r1.compose(r2)
    shift = r1.order()
    composed_initial_values = Matrix.hstack(
        initial_values,
        Matrix([solution[:shift]]),
    )
    expected = solution[shift:]
    actual = rr.evaluate_solution(composed_initial_values, start + shift, end)

    assert expected == actual


def test_fold_is_compose():
    r = LinearRecurrence([n**2 + 3 * n, 5 * n - 7, 13 * n**3, 2])
    multiplier = 12 * n**3 - 34 * n**2 + 56 * n - 78
    assert r.fold(multiplier) == LinearRecurrence([1, multiplier]).compose(r)


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
    assert 1 == len(unfolded)
    pcf = PCF(unfolded[0].recurrence_matrix).deflate_all()
    assert PCF(-2 * (n + 2), -((n + 1) ** 2)) == pcf


def test_euler_gompertz_independent():
    # from https://arxiv.org/abs/0902.1768
    r = LinearRecurrence(
        [
            -(16 * n + 1) * (16 * n - 15),
            (16 * n - 15) * (256 * n**3 + 528 * n**2 + 352 * n + 73),
            -(16 * n + 17) * (128 * n**3 + 40 * n**2 - 82 * n - 45),
            n**2 * (16 * n + 17) * (16 * n + 1),
        ]
    )

    def gompertz(mp):
        return -mp.e * mp.ei(-1)

    def constant(mp):
        return mp.e * mp.euler + gompertz(mp)

    limit = r.limit(200, 1)
    assert Matrix([[0, 0, 17], [-3, -2, 7]]) == limit.identify(constant(limit.mp))
