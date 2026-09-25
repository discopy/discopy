from pytest import raises

from discopy.traced import *
from discopy.utils import AxiomError


def test_trace_repr():
    assert repr(Box('f', 'x', 'x').trace()) == "traced.Trace(f, left=False)"


def test_trace_error():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_trace_dagger():
    f = Box('f', 'x', 'x')
    assert f.trace().dagger() == f.dagger().trace()


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


def test_Sum():
    f = Box('f', 'x', 'x')
    assert Sum([f]) == f
    assert isinstance(f + f, Sum) and (f + f).terms == (f, f)
