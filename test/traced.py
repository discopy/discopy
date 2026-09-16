from discopy import feedback
from discopy.traced import *


def test_trace_repr():
    assert repr(Box('f', 'x', 'x').trace()) == "traced.Trace(f, left=False)"


def test_trivial_delay_and_feedback():
    x = Ty('x')
    f = Box('f', x @ x, x @ x)
    assert isinstance(f, feedback.Diagram) and isinstance(x, feedback.Ty)
    assert f.d == f and x.d == x
    assert f.feedback() == f.trace()
    assert f.feedback(mem=x @ x) == f.trace(2)
    assert f.feedback(mem=Ty()) == f
    assert f.to_map().feedback() == f.to_map().trace()
    assert f.to_drawing().feedback() == f.to_drawing().trace()


def test_yanking_up_to_hypergraph():
    x = Ty('x')
    assert Equation(Swap(x, x).trace(), Id(x), Swap(x, x).trace(left=True))


def test_trace_vanishing():
    from discopy import compact, matrix, ribbon
    from discopy.python import additive, multiplicative

    x = compact.Ty('x')
    f = compact.Box('f', x @ x, x @ x)
    assert f.trace(0) == f
    assert f.to_hypergraph().trace(0) == f.to_hypergraph()
    assert f.to_map().trace(0) == f.to_map()
    assert f.to_drawing().trace(0) == f.to_drawing()

    y = ribbon.Ty('y')
    g = ribbon.Box('g', y @ y, y @ y)
    assert g.trace(0) == g

    assert matrix.Matrix[bool].swap(1, 1).trace(0)\
        == matrix.Matrix[bool].swap(1, 1)

    h = additive.Function(lambda i, tag=0: (i, tag), (int, int), (int, int))
    assert h.trace(0) == h

    k = multiplicative.Function(lambda i, j: (i, j), (int, int), (int, int))
    assert k.trace(0) == k
