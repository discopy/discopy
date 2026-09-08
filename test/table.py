import numpy as np
from pytest import mark, raises

from discopy import frobenius, table
from discopy.frobenius import (
    Box, Cap, Carrier, Cup, Diagram, Id, Spider, Swap, Ty)
from discopy.table import Morphism, Shard, SymbolTable, Wires
from discopy.utils import AxiomError

x, y = Ty('x'), Ty('y')
f, g, h = Box('f', x, y), Box('g', y, x), Box('h', x, x)
k = Box('k', x, x)
state = Box('s', Ty(), x)


def test_grow():
    assert table.grow(np.zeros(0, dtype=int), 1).shape == (1, )
    assert table.grow(np.zeros((2, 3), dtype=int), 3).shape == (4, 3)


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


def test_Carrier_repr():
    carrier = Carrier([x, x], [(f, 0, 1)], [(0, 1)])
    assert carrier == eval(repr(carrier)) != Carrier()
    assert carrier.cells == ((f, 0, 1), )


def test_Carrier_intern():
    carrier = Carrier()
    a = carrier.wires(x)
    assert carrier.intern(f, a.inside) == carrier.intern(f, a.inside)
    assert len(carrier.rows) == 1
    assert carrier.intern(h, a.inside) != carrier.intern(f, a.inside)


def test_Carrier_rebuild():
    carrier = Carrier()
    a, b = carrier.wires(x), carrier.wires(x)
    u, v = carrier.intern(f, a.inside), carrier.intern(f, b.inside)
    assert carrier.uf.find(u[0]) != carrier.uf.find(v[0])
    carrier.merge(a.inside[0], b.inside[0])
    carrier.rebuild()
    assert carrier.uf.find(u[0]) == carrier.uf.find(v[0])
    assert len(carrier.dead) == 1
    assert list(carrier.scan())[0][0] == 0


def test_Wires():
    carrier = Carrier()
    a, b = carrier.wires(x), carrier.wires(y)
    assert (a @ b).ty == x @ y and len(a @ b) == 2
    assert a != b and a == Wires(carrier, a.inside, x)
    assert a != Wires(Carrier(), a.inside, x) and a != a.inside
    with raises(AxiomError):
        a @ Carrier().wires(x)


def test_Morphism_laws():
    carrier = Carrier()
    u, v, w = map(carrier.from_box, (f, g, h))
    assert Morphism.id(u.dom).then(u).equiv(u)
    assert u.then(Morphism.id(u.cod)).equiv(u)
    assert u.then(v).then(w).equiv(u.then(v.then(w)))
    assert (u @ v).equiv(Morphism(u.dom @ v.dom, u.cod @ v.cod))
    assert not u.equiv(v) and not u.equiv(Carrier().from_box(f))
    assert u == Morphism(u.dom, u.cod) != v and u != u.dom
    with raises(AxiomError):
        u.then(u)


def test_Morphism_extraction():
    carrier = Carrier()
    a = carrier.wires(x)
    cheap = carrier.intern(f, a.inside)
    costly = carrier.intern(f, carrier.intern(g, carrier.intern(f, a.inside)))
    carrier.merge(cheap[0], costly[0])
    carrier.rebuild()
    morphism = Morphism(a, Wires(carrier, cheap, y))
    assert morphism.to_diagram() == f


def test_Carrier_costs_cycle():
    carrier = Carrier()
    a = carrier.wires(x)
    carrier.merge(a.inside[0], carrier.intern(h, a.inside)[0])
    assert carrier.costs() == ({}, {carrier.uf.find(a.inside[0]): 0})


def test_Carrier_order():
    carrier = Carrier([x, x, x], [(h, 1, 2), (h, 0, 1)])
    morphism = Morphism(Wires(carrier, (0, ), x), Wires(carrier, (2, ), x))
    assert morphism.to_diagram() == h >> h
    cyclic = Carrier([x, x], [(h, 0, 1), (h, 1, 0)])
    with raises(AxiomError):
        cyclic.order({0, 1}, {1: 0, 0: 1})


def test_Morphism_then_closes_a_loop():
    carrier = Carrier()
    morphism = carrier.from_box(h)
    with raises(AxiomError):
        morphism.then(morphism)
    assert not morphism.equiv(Morphism.id(morphism.dom))


def test_Carrier_depends_on_diamond():
    carrier = Carrier()
    a, = carrier.wires(x).inside
    left, right = carrier.intern(h, (a, )), carrier.intern(k, (a, ))
    top, = carrier.intern(Box('t', x @ x, x), left + right)
    assert carrier.depends_on(top, a) and not carrier.depends_on(a, top)


def test_Morphism_state_lift():
    morphism = Carrier.from_diagram(state @ state)
    assert len(morphism.carrier.rows) == 2
    morphism.carrier.rebuild()
    assert len(morphism.carrier.dead) == 1
    assert len(morphism.to_hypergraph().boxes) == 2


@mark.parametrize("diagram", [
    f, f >> g, f @ g, Id(x), Swap(x, y), Cap(x, x) >> Cup(x, x),
    Id(x) @ Cap(x, x) >> Cup(x, x) @ Id(x), Spider(2, 1, x),
    Spider(1, 1, x, .5), Spider(0, 0, x), state @ state, state >> f,
    f @ Id(x) >> Id(y) @ h, (state @ state) >> (h @ h)])
def test_round_trip(diagram):
    morphism = Carrier.from_diagram(diagram)
    assert morphism.to_diagram().to_hypergraph() == diagram.to_hypergraph()


def test_from_diagram_generic():
    assert table.Carrier.from_diagram(f).carrier.category == frobenius.Diagram
