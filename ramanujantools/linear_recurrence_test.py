from sympy.abc import n

from ramanujantools import LinearRecurrence, Matrix


def test_repr():
    expected = "LinearRecurrence([n, 1, 3 - n**2])"
    r = eval(expected)
    assert expected == repr(r)


def test_relation():
    expected = [1, n, n**2, n**3 - 7, 13 * n - 12]
    assert expected == LinearRecurrence(expected).relation()


def test_matrix():
    relation = [1, n, n + 1]
    assert Matrix([[0, n + 1], [1, n]]) == LinearRecurrence(relation).recurrence_matrix


def test_limit():
    r = LinearRecurrence([1, n, n**2])
    depths = [2, 3, 19, 101]
    assert r.limit(depths) == r.recurrence_matrix.limit({n: 1}, depths, {n: 1})


def test_compose():
    r = LinearRecurrence([1, n, n**2 - 3])
    assert LinearRecurrence(
        [1, n - 1, n**2 - 3 + n - 1, (n - 1) ** 2 - 3]
    ) == r.compose({n: 1})
