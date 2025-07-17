from sympy.abc import z

from ramanujantools.cmf import MeijerG


def test_meijer_g_conserving():
    for p in range(1, 3):
        for q in range(1, 3):
            for n in range(p):
                for m in range(q):
                    print(m, n, p, q)
                    MeijerG(m, n, p, q, z).assert_conserving()
