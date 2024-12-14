from sympy.abc import n
import sympy as sp

from ramanujantools import IntegerRelation, LinearRecurrence, Matrix
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
    r = LinearRecurrence([1, n, n**2])
    depths = [2, 3, 19, 101]
    assert r.limit(depths) == r.recurrence_matrix.limit({n: 1}, depths, {n: 1})


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


def test_compose():
    r = LinearRecurrence([f("a"), f("b"), f("c")])
    assert LinearRecurrence(
        [
            f("a"),
            -f("a", n - 1) * f("d") + f("b"),
            f("b", n - 1) * f("d") + f("c"),
            f("c", n - 1) * f("d"),
        ]
    ) == r.compose(f("d"))


def test_decompose_poly():
    r = LinearRecurrence([1, n, n])
    composition = n - 3
    composed = r.compose(composition)
    assert r == composed.decompose_poly(composition)


def test_decompose():
    r = LinearRecurrence([1, n, (n - 3) * (n + 7) * (n - 11)])
    composition = (n - 1) * (n + 2)
    composed = r.compose(composition)
    assert r == composed.decompose_poly(composition)


def test_euler():
    # from https://arxiv.org/abs/1010.1420
    m = Matrix([[0, -(n**2), 0], [1, 2 * n + 2, 0], [0, -(n - 1) / (n + 1), 1]])
    r = LinearRecurrence(m)
    assert [] == r.decompose()
    lim = r.limit(1000, 2)
    assert IntegerRelation([[-3, -12, -59], [6, 21, 102]]) == lim.identify(lim.mp.euler)
    assert IntegerRelation([[1, 4, 20], [-2, -7, -34]]) == lim.identify(
        -lim.mp.e * lim.mp.ei(-1)  # Gompertz
    )
    inflated = r.inflate(2 - n)

    lim = inflated.limit(1000, 5)
    assert IntegerRelation(
        [[-3625, 26792, -461718], [6270, -46380, 799620]]
    ) == lim.identify(lim.mp.euler, maxcoeff=10**6)

    decomposed = inflated.decompose_poly(n)
    pcf = decomposed.recurrence_matrix.as_pcf().pcf
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
