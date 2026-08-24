# -*- coding: utf-8 -*-
"""B28: Stream.permutation is broken for non-identity permutations (discopy/stream.py:537).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy import stream, symmetric


def test_b28_swap_permutation():
    x = stream.Ty.sequence(symmetric.Ty('x'))
    y = stream.Ty.sequence(symmetric.Ty('y'))
    s = stream.Stream.permutation((1, 0), [x, y])
    assert s.dom.now == x.now @ y.now
    assert s.cod.now == y.now @ x.now
    assert s.now.dom == x.now @ y.now and s.now.cod == y.now @ x.now
