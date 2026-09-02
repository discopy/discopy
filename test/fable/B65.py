"""B65: CMap.trace never type-checks the traced wires (discopy/cmap.py:1067).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy.compact import Box, Ty
from discopy.utils import AxiomError

x, y = Ty('x'), Ty('y')


def test_b65_ill_typed_trace_raises():
    with pytest.raises(AxiomError):
        Box('f', x, y).to_map().trace()


def test_b65_diagram_and_hypergraph_raise_control():
    """Passing control: the two other representations already refuse it."""
    with pytest.raises(AxiomError):
        Box('f', x, y).trace()
    with pytest.raises(AxiomError):
        Box('f', x, y).to_hypergraph().trace()


def test_b65_well_typed_trace_still_builds_control():
    """Passing control: matching types are not what the check rejects."""
    g = Box('g', x, x)
    assert g.to_map().trace().to_diagram() == g.trace().to_map().to_diagram()
