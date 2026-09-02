"""B75: make_causal types the cut wire with the unwound spider type, so a trace over an adjoint-typed wire cannot be decoded (discopy/hypergraph.py:1309).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np

from discopy import compact
from discopy.tensor import Tensor

x, y = map(compact.Ty, "xy")
g = compact.Box('g', x @ y.r, x @ y.r)
F = compact.Functor(
    ob_map={x: 2, y: 3}, ar_map={g: np.arange(36.)}, cod=Tensor)


def test_b75_trace_over_adjoint_wire_decodes():
    decoded = g.trace().to_hypergraph().to_diagram()
    assert np.allclose(F(decoded).array, F(g.trace()).array)


def test_b75_left_trace_over_adjoint_wire_simplifies():
    g2 = compact.Box('g2', y.r @ x, y.r @ x)
    assert g2.trace(left=True).simplify().cod == x
