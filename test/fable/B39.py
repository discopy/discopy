"""B39: repr transparency violations across traced, cat, interaction, channel and quantum (discopy/traced.py:208, cat.py:788, interaction.py:157, quantum/channel.py:96, quantum/gates.py).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import discopy
from discopy import cat, frobenius, interaction, traced
from discopy.quantum import Measure
from discopy.quantum.channel import C
from discopy.tensor import Dim


def test_b39_traced_trace_repr():
    x = traced.Ty('x')
    trace = traced.Box('f', x @ x, x @ x).trace(left=False)
    assert eval(repr(trace), {'traced': traced}) == trace


def test_b39_cat_bubble_repr_keeps_kwargs():
    bubble = cat.Bubble(
        cat.Box('f', cat.Ob('x'), cat.Ob('y')), method='grad', name='B')
    assert eval(repr(bubble), {'cat': cat}) == bubble


def test_b39_interaction_diagram_repr():
    T = frobenius.Ty
    X = interaction.Ty[T](T('x0'), T('x1'))
    Y = interaction.Ty[T](T('y0'), T('y1'))
    D = interaction.Diagram[frobenius.Diagram]
    f = D(frobenius.Box('f', T('x0') @ T('y1'), T('y0') @ T('x1')), X, Y)
    namespace = {'interaction': interaction, 'frobenius': frobenius,
                 'Diagram': interaction.Diagram, 'cat': cat}
    assert eval(repr(f), namespace) == f


def test_b39_channel_classical_str():
    assert str(C(Dim(2))).startswith('C(')


def test_b39_quantum_measure_repr():
    assert eval(repr(Measure()), {'quantum': discopy.quantum}) == Measure()
