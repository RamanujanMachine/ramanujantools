from pytest import approx
from sympy.abc import c, n
import mpmath as mp

from ramanujan.pcf import PCF

def test_delta_sequence():
    pcf = PCF(2*n+1, n**2)
    lim = 4/mp.pi
    depths = [1,20,200]
    
    test_deltas = pcf.delta_sequence(depths, limit=lim)
    true_deltas = []
    for dep in depths:
        true_deltas.append(pcf.delta(dep, lim))

    assert test_deltas == true_deltas


# ADD: apery's test - known CMF, test_apery
