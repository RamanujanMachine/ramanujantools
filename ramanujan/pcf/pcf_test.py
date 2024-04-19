from pytest import approx
from sympy.abc import c, n
import mpmath as mp

from ramanujan.pcf import PCF

def test_repr():
    pcf = PCF(1 + n, 3 - n)
    assert pcf == eval(repr(pcf))


def test_degree():
    pcf = PCF(1 + n - n**2, 3 - n**9)
    assert (2, 9) == pcf.degree()


def test_limit():
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    expected = (4 ** (1 / 3) + 1) / (4 ** (1 / 3) - 1)
    assert expected == approx(pcf.limit(100).ratio(), 1e-4)


def test_walk_list():
    iterations = [1, 2, 3, 17, 29, 53, 99]
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    assert pcf.walk(iterations) == [pcf.walk(i) for i in iterations]


def test_limit_list():
    iterations = [1, 2, 3, 17, 29, 53, 99]
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    assert pcf.limit(iterations) == [pcf.limit(i) for i in iterations]


def test_inflate_constant():
    a_n = n + 4
    b_n = n**2
    pcf = PCF(a_n, b_n)
    assert PCF(c * a_n, c**2 * b_n) == pcf.inflate(c)


def test_inflate_poly():
    a_n = n + 4
    b_n = n**2
    c_n = n**7 + 5 * n - 3
    pcf = PCF(a_n, b_n)
    assert PCF(c_n * a_n, c_n.subs(n, n - 1) * c_n * b_n) == pcf.inflate(c_n)


def test_deflate_constant():
    a_n = n + 4
    b_n = n**2
    pcf = PCF(c * a_n, c**2 * b_n)
    assert PCF(a_n, b_n) == pcf.deflate(c)


def test_deflate_poly():
    a_n = n + 4
    b_n = n**2
    c_n = n**7 + 5 * n - 3
    pcf = PCF(c_n * a_n, c_n.subs(n, n - 1) * c_n * b_n)
    assert PCF(a_n, b_n) == pcf.deflate(c_n)


def test_deflate_all():
    c_n = c**2 * (7 * n - 13 * c) * (2 * n - 5) ** 4 * (3 * n + 11)
    a_n = n + c
    b_n = n**2 - c * n
    pcf = PCF(c_n * a_n, c_n.subs(n, n - 1) * c_n * b_n)
    assert PCF(a_n, b_n) == pcf.deflate_all()


def test_blind_delta():
    mp.mp.dps = 10**5
    pcf = PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))
    depth = 2000
    delta = pcf.delta(depth)
    assert delta > 0.086 and delta < 0.087


def test_precision_e():
    pcf = PCF(n, n)
    assert PCF.precision(pcf.walk(2**10)) == 2642


def test_precision_phi():
    pcf = PCF(1, 1)
    assert PCF.precision(pcf.walk(2**10)) == 427


def test_delta_sequence_agrees_with_delta():
    pcf = PCF(2*n+1, n**2)
    limit = 4/mp.pi
    depth = 50
    
    actual_deltas = pcf.delta_sequence(depth, limit)
    expected_deltas = []
    for dep in range(1, depth + 1):
        expected_deltas.append(pcf.delta(dep, limit))

    assert expected_deltas == actual_deltas


def test_blind_delta_sequence_agrees_with_blind_delta():
    pcf = PCF(2*n+1, n**2)
    depth = 50
    mlim = pcf.limit(2 * depth)
    limit = mlim.ratio()
    
    actual_values = pcf.delta_sequence(depth)
    expected_deltas = []
    for dep in range(1, depth + 1):
        expected_deltas.append(pcf.delta(dep, limit))

    assert expected_deltas == actual_values
