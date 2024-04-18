from pytest import approx
from sympy.abc import c, n
import mpmath as mp

from ramanujan.pcf import PCF 

def test_delta_sequence():
    pcf = PCF(2*n+1, n**2)
    lim = 4/mp.pi
    depths = [1, 20, 100, 200]
    
    test_deltas = pcf.delta_sequence(depths, limit=lim)
    true_deltas = []
    for dep in depths:
        true_deltas.append(pcf.delta(dep, lim))

    # assert np.average(np.abs(true_deltas - test_deltas)) < 0.001
    print(f"Test deltas: {test_deltas}")
    print(f"True deltas: {true_deltas}")
    # np.array(test_deltas) == approx(np.array(true_deltas), 1e-3)
    assert test_deltas == true_deltas


if __name__ == "main":
    test_delta_sequence()
