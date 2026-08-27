from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_to_hypergraph():
    x, y = map(Ty, "xy")
    f = Box('f', x, y)
    assert f.transpose().to_hypergraph() == f.to_hypergraph().transpose()


def test_to_hypergraph_rejects_non_boundary_connected_diagrams():
    x, y = map(Ty, "xy")
    circle = lambda typ: Cap(typ, typ.r) >> Cup(typ, typ.r)
    nested = Cap(x, x.r) >> x @ circle(y) @ x.r >> Cup(x, x.r)
    side_by_side = circle(x) @ circle(y)
    assert nested != side_by_side
    assert Hypergraph.from_diagram(nested) == Hypergraph.from_diagram(
        side_by_side)
    with raises(NotImplementedError):
        side_by_side.to_hypergraph()


def test_strategy():
    from hypothesis import find

    from discopy import testing

    testing.assert_strategy_finds(Diagram, Cup, Cap)
    winding = find(Ty.strategy(min_length=1),
                   lambda value: value.inside[0].z)
    assert winding.inside[0].z == 1


def test_axioms():
    from discopy import testing

    testing.assert_axioms(Ty, Diagram, Functor)
