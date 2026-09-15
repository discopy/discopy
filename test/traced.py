from discopy import feedback
from discopy.traced import *


def test_trace_repr():
    assert repr(Box('f', 'x', 'x').trace()) == "traced.Trace(f, left=False)"


def test_trivial_delay_and_feedback():
    x = Ty('x')
    f = Box('f', x @ x, x @ x)
    assert isinstance(f, feedback.Diagram) and isinstance(x, feedback.Ty)
    assert f.delay() == f and x.delay() == x
    assert f.feedback() == f.trace()
    assert f.feedback(mem=x @ x) == f.trace(2)
    assert f.feedback(mem=Ty()) == f
    assert f.to_map().feedback() == f.to_map().trace()
    assert f.to_drawing().feedback() == f.to_drawing().trace()


def test_yanking_up_to_hypergraph():
    x = Ty('x')
    assert Equation(Swap(x, x).trace(), Id(x), Swap(x, x).trace(left=True))
