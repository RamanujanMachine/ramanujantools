from pytest import approx
from sympy.abc import c, n
import mpmath as mp

import ramanujantools as rt
from ramanujantools.pcf import PCF


def test_repr():
    pcf = PCF(1 + n, 3 - n)
    assert pcf == eval(repr(pcf))


def test_degree():
    pcf = PCF(1 + n - n**2, 3 - n**9)
    assert (2, 9) == pcf.degree()


def test_singular_points():
    a_n = n + 1
    b_n = (n + 1) * (n - 17) * (n + 59) * (n - 102)
    pcf = PCF(a_n, b_n)
    assert pcf.singular_points() == [{n: 17}, {n: 102}]


def test_limit_as_float():
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    expected = (4 ** (1 / 3) + 1) / (4 ** (1 / 3) - 1)
    assert expected == approx(pcf.limit(100).as_float(), 1e-4)


def test_walk_list():
    iterations = [1, 2, 3, 17, 29, 53, 99]
    pcf = PCF(5 + 10 * n, 1 - 9 * n**2)
    assert pcf.walk(iterations) == [pcf.walk(i) for i in iterations]


def test_walk_start():
    iterations = [1]
    p = PCF(n + 7, 3 * n**2 - 1)
    expected = p.walk(sum(iterations))
    actual = rt.Matrix.eye(2)
    for i in range(len(iterations)):
        actual *= p.walk(iterations[i], start=sum(iterations[0:i]))
    assert expected == actual


def test_inflate_constant():
    c = 3
    a_n = n + 4
    b_n = n**2
    pcf = PCF(a_n, b_n)
    assert PCF(c * a_n, c**2 * b_n) == pcf.inflate(c)


def test_inflate_symbol():
    a_n = n + 4
    b_n = n**2
    pcf = PCF(a_n, b_n)
    assert PCF(c * a_n, c**2 * b_n) == pcf.inflate(c)


def test_inflate_poly():
    a_n = n + 4
    b_n = n**2
    c_n = n**7 + 5 * n - 3
    pcf = PCF(a_n, b_n)
    assert PCF(c_n * a_n, c_n.subs({n: n - 1}) * c_n * b_n) == pcf.inflate(c_n)


def test_deflate_constant():
    a_n = n + 4
    b_n = n**2
    pcf = PCF(c * a_n, c**2 * b_n)
    assert PCF(a_n, b_n) == pcf.deflate(c)


def test_deflate_poly():
    a_n = n + 4
    b_n = n**2
    c_n = n**7 + 5 * n - 3
    pcf = PCF(c_n * a_n, c_n.subs({n: n - 1}) * c_n * b_n)
    assert PCF(a_n, b_n) == pcf.deflate(c_n)


def test_deflate_all():
    c_n = c**2 * (7 * n - 13 * c) * (2 * n - 5) ** 4 * (3 * n + 11)
    a_n = n + c
    b_n = n**2 - c * n
    pcf = PCF(c_n * a_n, c_n.subs({n: n - 1}) * c_n * b_n)
    assert PCF(a_n, b_n) == pcf.deflate_all()


def test_blind_delta():
    pcf = PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))
    depth = 2000
    delta = pcf.delta(depth)
    assert delta > 0.08


def test_precision_e():
    pcf = PCF(n, n)
    assert pcf.limit(2**10 + 1).precision() == 2642


def test_precision_phi():
    pcf = PCF(1, 1)
    assert pcf.limit(2**10 + 1).precision() == 427


def test_delta_sequence_agrees_with_delta():
    # To prevent dynamic precision errors, setting it high enough
    mp.mp.dps = 50
    pcf = PCF(2 * n + 1, n**2)
    depth = 50
    limit = 4 / mp.pi

    actual_deltas = pcf.delta_sequence(depth, limit)
    expected_deltas = []
    for dep in range(1, depth + 1):
        expected_deltas.append(pcf.delta(dep, limit))

    assert expected_deltas == actual_deltas


def test_blind_delta_sequence_agrees_with_blind_delta():
    # To prevent dynamic precision errors, setting it high enough
    mp.mp.dps = 50
    pcf = PCF(2 * n + 1, n**2)
    depth = 50
    limit = pcf.limit(2 * depth).as_float()

    actual_values = pcf.delta_sequence(depth)
    expected_deltas = []
    for dep in range(1, depth + 1):
        expected_deltas.append(pcf.delta(dep, limit))

    assert expected_deltas == actual_values


def test_as_latex_not_generic():
    pcf = PCF(1 + n - n**2, 3 - n**9)

    expected_depth5_start1 = (
        '1 + \\cfrac{2}{1 + \\cfrac{-509}{-1 + \\cfrac{-19680}'
        '{-5 + \\cfrac{-262141}{-11 + \\cfrac{-1953122}'
        '{\\ddots + \\cfrac{3 - n^9}{-n^2 + n + 1 + \\ddots}}}}}}')
    expected_depth7_start3 = (
        '\\cfrac{-19680}{-5 + \\cfrac{-262141}{-11 + \\cfrac{-1953122}'
        '{-19 + \\cfrac{-10077693}{-29 + \\cfrac{-40353604}'
        '{\\ddots + \\cfrac{3 - n^9}{-n^2 + n + 1 + \\ddots}}}}}}')
    
    assert expected_depth5_start1 == pcf.as_latex(depth=5, start=1)
    assert expected_depth7_start3 == pcf.as_latex(depth=7, start=3)


def test_as_latex_generic():

    expected_depth5_start1 = (
        'a_0 + \\cfrac{b_1}{a_1 + \\cfrac{b_2}{a_2 + \\cfrac{b_3}'
        '{a_3 + \\cfrac{b_4}{a_4 + \\cfrac{b_5}'
        '{\\ddots + \\cfrac{b_n}{a_n + \\ddots}}}}}}')
    expected_depth7_start3 = (
        '\\cfrac{b_3}{a_3 + \\cfrac{b_4}'
        '{a_4 + \\cfrac{b_5}{a_5 + \\cfrac{b_6}{a_6 + \\cfrac{b_7}'
        '{\\ddots + \\cfrac{b_n}{a_n + \\ddots}}}}}}')
    
    assert expected_depth5_start1 == PCF.as_latex(depth=5, start=1)
    assert expected_depth7_start3 == PCF.as_latex(depth=7, start=3)

