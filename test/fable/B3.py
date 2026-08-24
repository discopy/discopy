"""B3: trace is unusable under default type checking (discopy/python/multiplicative.py:200).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.python import multiplicative


def test_b3_trace_call():
    f = multiplicative.Function(
        lambda x, y=0: (x, y), (int, int), (int, int))
    assert f.trace()(5) == 5


def test_b3_trace_two_wires_builds():
    f = multiplicative.Function(
        lambda x, y=0, z=0: (x, y, z), (int,) * 3, (int,) * 3)
    traced = f.trace(2)
    assert traced.dom == (int,) and traced.cod == (int,)
