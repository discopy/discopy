"""B71: to_diagram puts every domain-less box at the right of the scan (discopy/cmap.py:1455).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy import monoidal, pivotal, rigid, symmetric
from discopy.symmetric import Trace


@pytest.mark.parametrize("module", [monoidal, rigid, pivotal, symmetric])
def test_b71_state_left_of_a_wire_decodes_where_it_was(module):
    x = module.Ty('x')
    s = module.Box('s', module.Ty(), x)
    assert (s @ x).to_hypergraph().to_diagram() == s @ x
    assert (s @ x).to_map().to_diagram() == s @ x


def test_b71_rigid_transpose_round_trips():
    x, y = rigid.Ty('x'), rigid.Ty('y')
    transposed = rigid.Box('f', x, y).transpose()
    assert transposed.to_map().to_diagram() == transposed


def test_b71_pivotal_cap_left_of_a_wire_round_trips():
    x = pivotal.Ty('x')
    diagram = pivotal.Cap(x, x.r) @ x
    rightmost = x @ pivotal.Cap(x, x.r)
    assert rightmost.to_map().to_diagram() == rightmost  # already passes
    assert diagram.to_map().to_diagram() == diagram


def _out_of_order():
    """g >> f stored as boxes (f, g): a backward wire closing no cycle."""
    x = symmetric.Ty('x')
    f, g = symmetric.Box('f', x, x), symmetric.Box('g', x, x)
    return symmetric.CMap(x, x, (f, g), [3, 4, 5, 0, 1, 2]), f, g


def test_b71_make_causal_reorders_without_a_loop_control():
    """Passing control: with no loop the backward wire is reordered."""
    unordered, f, g = _out_of_order()
    assert unordered.make_causal().boxes == (g, f)


def test_b71_make_causal_reorders_beside_a_scalar_loop():
    unordered, f, g = _out_of_order()
    loop = symmetric.CMap(
        symmetric.Ty(), symmetric.Ty(), (), (), loops=(symmetric.Ty('x'), ))
    causal = (unordered @ loop).make_causal()
    traces = [box for box in causal.boxes if isinstance(box, Trace)]
    assert len(traces) == 1, causal.boxes
    assert not any(isinstance(box, Trace) for box in traces[0].arg.boxes), \
        "the merely backward f -> g wire was cut into a second trace"
