"""B78: simplify dies on a foliated layer and to_braided on a balanced Trace (discopy/braided.py:109, balanced.py:343).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy import balanced, braided, traced


def test_b78_simplify_on_single_box_layers_control_passes():
    x, y = braided.Ty('x'), braided.Ty('y')
    f, g = braided.Box('f', x, x), braided.Box('g', y, y)
    braid = braided.Braid(x, y)
    diagram = f @ y >> x @ g >> braid >> braid[::-1]
    assert diagram.simplify() == f @ y >> x @ g


def test_b78_simplify_after_foliation():
    x, y = braided.Ty('x'), braided.Ty('y')
    f, g = braided.Box('f', x, x), braided.Box('g', y, y)
    braid = braided.Braid(x, y)
    diagram = (f @ g >> braid >> braid[::-1]).foliation()
    assert diagram.simplify() == (f @ g).foliation()


def test_b78_balanced_trace_to_braided():
    z = balanced.Ty('z')
    diagram = balanced.Box('g', z @ z, z @ z).trace().to_braided()
    assert diagram.dom == diagram.cod
    assert isinstance(diagram.boxes[0], traced.Trace)
