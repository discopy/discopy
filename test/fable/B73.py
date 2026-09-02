"""B73: foliation and depth crash on a Trace beside another box (discopy/monoidal.py:1099).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy import symmetric, traced


def _boxes(module):
    z = module.Ty('z')
    return module.Box('g', z @ z, z @ z).trace(), module.Box('f', z, z)


def test_b73_symmetric_foliation_trace_then_box():
    trace, f = _boxes(symmetric)
    assert (trace >> f).foliation() == trace >> f


def test_b73_symmetric_foliation_trace_beside_box():
    trace, f = _boxes(symmetric)
    foliated = (trace @ f).foliation()
    assert len(foliated) == 1 and foliated.boxes == [trace, f]


def test_b73_traced_foliation_trace_then_box():
    trace, f = _boxes(traced)
    assert (trace >> f).foliation() == trace >> f


def test_b73_traced_depth():
    trace, f = _boxes(traced)
    assert (trace >> f).depth() == 2 and (trace @ f).depth() == 1
