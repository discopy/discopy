"""B2: trace feeds back on the wrong wire when len(dom') != len(cod') (discopy/python/additive.py:109).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.python import additive


def test_b2_trace_feedback_wire():
    # f has dom (int, int, int) and cod (int, int); trace() feeds output
    # tag 1 (the traced output) back into input tag 2 (the traced input).
    def inside(obj, tag):
        if tag == 0:
            return (obj, 1)  # send to the traced output
        if tag == 1:
            return (obj * 1000, 0)  # marker: wrongly re-fed at input tag 1
        return (obj + 5, 0)  # the traced input, tag 2
    f = additive.Function(inside, (int, int, int), (int, int))
    assert f.trace()(7, 0) == 12
