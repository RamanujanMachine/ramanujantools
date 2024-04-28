import mpmath as mp

from ramanujantools import delta


def test_delta():
    assert delta(22, 7, mp.pi) > 2
