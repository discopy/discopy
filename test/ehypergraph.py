import numpy as np
from pytest import mark, raises

from discopy import frobenius, ehypergraph
from discopy.frobenius import (
    Box, Cap, Cup, Diagram, EHypergraph, Id, Spider, Swap, Ty)
from discopy.ehypergraph import Morphism, Shard, SymbolTable, Wires
from discopy.utils import AxiomError

x, y = Ty('x'), Ty('y')
f, g, h = Box('f', x, y), Box('g', y, x), Box('h', x, x)
k = Box('k', x, x)
state = Box('s', Ty(), x)


def test_grow():
    assert ehypergraph.grow(np.zeros(0, dtype=int), 1).shape == (1, )
    assert ehypergraph.grow(np.zeros((2, 3), dtype=int), 3).shape == (4, 3)


def test_SymbolTable():
    symbols = SymbolTable([1])
    assert symbols.intern(True) == 1 != symbols.intern(1)
    assert symbols[0] == 1 and len(symbols) == 2
    assert symbols == eval(repr(symbols)) != SymbolTable()


def test_Shard():
    shard = Shard(1, 1, [(0, 1)])
    for row in [(2, 3), (4, 5)]:
        shard.append(row)
    assert list(shard) == [(0, 1), (2, 3), (4, 5)] and len(shard) == 3
    assert shard.src.tolist() == [[0], [2], [4]]
    assert shard.tgt.tolist() == [[1], [3], [5]]
    assert shard == eval(repr(shard)) != Shard(1, 1)
    with raises(ValueError):
        shard.append((0, 1, 2))


def test_EHypergraph_repr():
    graph = EHypergraph([x, x], [(f, 0, 1)], [(0, 1)])
    assert graph == eval(repr(graph)) != EHypergraph()
    assert graph.cells == ((f, 0, 1), )


def test_EHypergraph_intern():
    graph = EHypergraph()
    a = graph.wires(x)
    assert graph.intern(f, a.inside) == graph.intern(f, a.inside)
    assert len(graph.rows) == 1
    assert graph.intern(h, a.inside) != graph.intern(f, a.inside)


def test_EHypergraph_rebuild():
    graph = EHypergraph()
    a, b = graph.wires(x), graph.wires(x)
    u, v = graph.intern(f, a.inside), graph.intern(f, b.inside)
    assert graph.uf.find(u[0]) != graph.uf.find(v[0])
    graph.merge(a.inside[0], b.inside[0])
    graph.rebuild()
    assert graph.uf.find(u[0]) == graph.uf.find(v[0])
    assert len(graph.dead) == 1
    assert list(graph.scan())[0][0] == 0


def test_Wires():
    graph = EHypergraph()
    a, b = graph.wires(x), graph.wires(y)
    assert (a @ b).ty == x @ y and len(a @ b) == 2
    assert a != b and a == Wires(graph, a.inside, x)
    assert a != Wires(EHypergraph(), a.inside, x) and a != a.inside
    with raises(AxiomError):
        a @ EHypergraph().wires(x)


def test_Morphism_laws():
    graph = EHypergraph()
    u, v, w = map(graph.from_box, (f, g, h))
    assert Morphism.id(u.dom).then(u).equiv(u)
    assert u.then(Morphism.id(u.cod)).equiv(u)
    assert u.then(v).then(w).equiv(u.then(v.then(w)))
    assert (u @ v).equiv(Morphism(u.dom @ v.dom, u.cod @ v.cod))
    assert not u.equiv(v) and not u.equiv(EHypergraph().from_box(f))
    assert u == Morphism(u.dom, u.cod) != v and u != u.dom
    with raises(AxiomError):
        u.then(u)


def test_Morphism_extraction():
    graph = EHypergraph()
    a = graph.wires(x)
    cheap = graph.intern(f, a.inside)
    costly = graph.intern(f, graph.intern(g, graph.intern(f, a.inside)))
    graph.merge(cheap[0], costly[0])
    graph.rebuild()
    morphism = Morphism(a, Wires(graph, cheap, y))
    assert morphism.to_diagram() == f


def test_EHypergraph_costs_cycle():
    graph = EHypergraph()
    a = graph.wires(x)
    graph.merge(a.inside[0], graph.intern(h, a.inside)[0])
    assert graph.costs() == ({}, {graph.uf.find(a.inside[0]): 0})


def test_EHypergraph_order():
    graph = EHypergraph([x, x, x], [(h, 1, 2), (h, 0, 1)])
    morphism = Morphism(Wires(graph, (0, ), x), Wires(graph, (2, ), x))
    assert morphism.to_diagram() == h >> h
    cyclic = EHypergraph([x, x], [(h, 0, 1), (h, 1, 0)])
    with raises(AxiomError):
        cyclic.order({0, 1}, {1: 0, 0: 1})


def test_Morphism_then_closes_a_loop():
    graph = EHypergraph()
    morphism = graph.from_box(h)
    with raises(AxiomError):
        morphism.then(morphism)
    assert not morphism.equiv(Morphism.id(morphism.dom))


def test_EHypergraph_depends_on_diamond():
    graph = EHypergraph()
    a, = graph.wires(x).inside
    left, right = graph.intern(h, (a, )), graph.intern(k, (a, ))
    top, = graph.intern(Box('t', x @ x, x), left + right)
    assert graph.depends_on(top, a) and not graph.depends_on(a, top)


def test_Morphism_state_lift():
    morphism = EHypergraph.from_diagram(state @ state)
    assert len(morphism.ehypergraph.rows) == 2
    morphism.ehypergraph.rebuild()
    assert len(morphism.ehypergraph.dead) == 1
    assert len(morphism.to_hypergraph().boxes) == 2


@mark.parametrize("diagram", [
    f, f >> g, f @ g, Id(x), Swap(x, y), Cap(x, x) >> Cup(x, x),
    Id(x) @ Cap(x, x) >> Cup(x, x) @ Id(x), Spider(2, 1, x),
    Spider(1, 1, x, .5), Spider(0, 0, x), state @ state, state >> f,
    f @ Id(x) >> Id(y) @ h, (state @ state) >> (h @ h)])
def test_round_trip(diagram):
    morphism = EHypergraph.from_diagram(diagram)
    assert morphism.to_diagram().to_hypergraph() == diagram.to_hypergraph()


def test_from_diagram_generic():
    morphism = ehypergraph.EHypergraph.from_diagram(f)
    assert morphism.ehypergraph.category == frobenius.Diagram
