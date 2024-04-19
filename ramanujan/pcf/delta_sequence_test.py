from pytest import approx
import sympy as sp
from sympy.abc import n, x, y
import mpmath as mp

from ramanujan.pcf import PCF
from ramanujan.cmf import CMF, known_cmfs
from mpmath import zeta


def test_delta_sequence_pi():
    pcf = PCF(2*n+1, n**2)
    lim = 4/mp.pi
    depth = 300
    
    test_deltas = pcf.delta_sequence(depth, limit=lim)
    true_deltas = []
    for dep in range(1, depth):
        true_deltas.append(pcf.delta(dep, lim))

    assert test_deltas == true_deltas


def test_delta_sequence_apery():
    mp.mp.dps = 5000

    # Apery's PCF
    pcf = PCF(34 * n**3 + 51 * n**2 + 27 * n + 5, -(n**6))

    limit = sp.Float(6 / zeta(3), mp.mp.dps)
    depth = 400
    deltas = pcf.delta_sequence(depth, limit)
    assert deltas[-1] > 0.086 and deltas[-1] < 0.87 
