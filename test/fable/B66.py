"""B66: Hypergraph.depth is longest_path // 4, two edges short when the path starts at a state (discopy/hypergraph.py:1627).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy import frobenius, symmetric

x = frobenius.Ty('x')
f, g = frobenius.Box('f', frobenius.Ty(), x), frobenius.Box('g', x, frobenius.Ty())
s = frobenius.Box('s', frobenius.Ty(), frobenius.Ty())


def test_b66_state_depth():
    assert f.depth() == 1


def test_b66_state_then_effect_depth():
    assert (f >> g).depth() == 2


def test_b66_scalar_depth():
    assert s.depth() == 1


def test_b66_symmetric_delegates():
    state = symmetric.Box('f', symmetric.Ty(), symmetric.Ty('x'))
    assert state.depth() == 1
