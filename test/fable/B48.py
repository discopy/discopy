"""B48: Hypergraph.rotate reverses the ports but never swaps a box's dom and cod, so any diagram with a rotated box is mis-encoded (discopy/hypergraph.py:494).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np

from discopy import compact, frobenius
from discopy.tensor import Tensor

x, y = map(compact.Ty, "xy")
f = compact.Box('f', x, y)


def test_b48_rotated_box_encodes():
    h = f.r.to_hypergraph()
    assert (h.dom, h.cod, h.boxes) == (y.r, x.r, (f.r, ))


def test_b48_rotate_is_an_involution():
    h = f.to_hypergraph()
    assert h.r.r == h


def test_b48_compact_equation_of_rotations():
    g = compact.Box('g', x @ y, x @ y)
    assert bool(compact.Equation(g.r, g.l))


def test_b48_frobenius_simplify_keeps_the_rotation():
    a = frobenius.Ty('a')
    fb = frobenius.Box('f', a, a)
    G = frobenius.Functor(
        {a: 2}, {fb: np.array([[1., 2.], [3., 4.]])}, cod=Tensor)
    assert np.array_equal(G(fb.r.simplify()).array, G(fb.r).array)
