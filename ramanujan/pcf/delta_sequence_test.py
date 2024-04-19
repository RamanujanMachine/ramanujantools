from pytest import approx
import sympy as sp
from sympy.abc import n, x, y
import mpmath as mp

from ramanujan.pcf import PCF
from ramanujan.cmf import CMF, known_cmfs
from mpmath import zeta


def test_delta_sequence_correct_values():
    pcf = PCF(2*n+1, n**2)
    limit = 4/mp.pi
    depth = 300
    
    test_deltas = pcf.delta_sequence(depth, limit)
    true_deltas = []
    for dep in range(1, depth + 1):
        true_deltas.append(pcf.delta(dep, limit))

    assert test_deltas == true_deltas


def test_delta_sequence_apery():
    mp.mp.dps = 5000
    # Apery's PCF
    pcf = PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))
    limit = sp.Float(6 / zeta(3), mp.mp.dps)
    depth = 400

    deltas = pcf.delta_sequence(depth, limit)
    assert deltas[-1] > 0.086 and deltas[-1] < 0.87


def test_delta_sequence_no_limit():
    mp.mp.dps = 10000
    pcf = PCF(6, (2*n+1)**2)
    limit = mp.pi + 3
    depth = 1000

    deltas = pcf.delta_sequence(depth)
    assert len(deltas) == depth
    assert deltas[-1] == approx(pcf.delta(depth, limit), 1e-1)

