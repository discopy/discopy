import numpy as np
from pytest import mark, raises

from discopy import compact, ehypergraph, frobenius, markov, rigid, symmetric
from discopy.ehypergraph import (
    Cartesian, Compact, Frobenius, Markov, Shard, SymbolTable, Traced)
from discopy.frobenius import Box, Cap, Cup, Diagram, Id, Spider, Swap, Ty
from discopy.utils import AxiomError

x, y = Ty('x'), Ty('y')
f, g, h = Box('f', x, y), Box('g', y, x), Box('h', x, x)
state = Box('s', Ty(), x)
F = Frobenius.from_diagram


def test_grow():
    assert ehypergraph.grow(np.zeros(0, dtype=int), 1).shape == (1, )
    assert ehypergraph.grow(np.zeros((2, 3), dtype=int), 3).shape == (4, 3)


def test_SymbolTable():
    symbols = SymbolTable([1])
    assert symbols.intern(True) == 1 != symbols.intern(1)
    assert symbols[0] == 1 and len(symbols) == 2
    assert symbols == eval(repr(symbols)) != SymbolTable()


def test_Shard():
    shard = Shard(h, [(0, 1)])
    for row in [(2, 3), (4, 5)]:
        shard.append(row)
    assert list(shard) == [(0, 1), (2, 3), (4, 5)] and len(shard) == 3
    assert shard.doms.tolist() == [[0], [2], [4]]
    assert shard.cods.tolist() == [[1], [3], [5]]
    assert shard == eval(repr(shard)) != Shard(h)
    assert Shard(state).doms.shape == (0, 0) and len(Shard(state)) == 0
    with raises(ValueError):
        shard.append((0, 1, 2))


def test_Shard_concatenate():
    left, right = Shard(h, [(0, 1)]), Shard(h, [(2, 3)])
    assert left.concatenate(right) == Shard(h, [(0, 1), (2, 3)])
    assert left.concatenate(Shard(h)) == left == Shard(h).concatenate(left)
    assert right.shift(2) == Shard(h, [(4, 5)]) != right
    grown = left.concatenate(right)
    grown.append((6, 7))
    assert list(left) == [(0, 1)] and len(grown) == 3
    with raises(ValueError):
        left.concatenate(Shard(f))


def test_structure():
    x_ = symmetric.Ty('x')
    assert Traced.swap(x_, x_).trace() == Traced.id(x_)
    assert Traced.swap(x_, x_).trace(left=True) == Traced.id(x_)
    assert Traced.swap(x_, x_).trace(0) == Traced.swap(x_, x_)
    loop = Traced.id(x_).trace()
    assert [box for _, box, *_ in loop.scan()] == [
        symmetric.Trace(symmetric.Id(x_))]
    x__ = compact.Ty('x')
    snake = x__ @ Compact.caps(x__.r, x__) >> Compact.cups(x__, x__.r) @ x__
    assert snake.equiv(Compact.id(x__))
    assert (Frobenius.spiders(1, 2, x) >> Frobenius.spiders(2, 1, x)).equiv(
        Frobenius.id(x))


def test_Frobenius_repr():
    graph = F(f >> g).then(F(h))
    graph.merge(*graph.dom_wires, *graph.cod_wires)
    assert graph == eval(repr(graph), {
        "ehypergraph": ehypergraph, "Diagram": Diagram,
        "frobenius": frobenius})
    assert graph != F(f >> g >> h)


def test_Frobenius_type_error():
    with raises(AxiomError):
        Frobenius(x, y, (f, ), ((0, ), (((0, ), (1, )), ), (1, )), [x, x])
    graph = Frobenius.id(x @ y)
    with raises(AxiomError):
        graph.merge(*graph.dom_wires)


def test_Frobenius_intern():
    graph = Frobenius.id(x)
    a, = graph.dom_wires
    assert graph.intern(f, (a, )) == graph.intern(f, (a, ))
    assert graph.intern(h, (a, )) != graph.intern(f, (a, ))
    assert len(graph.rows) == 2


def test_Frobenius_rebuild():
    graph = Frobenius.id(x @ x)
    a, b = graph.dom_wires
    (u, ), (v, ) = graph.intern(f, (a, )), graph.intern(f, (b, ))
    graph.merge(a, b)
    graph.rebuild()
    assert graph.uf.find(u) == graph.uf.find(v)
    assert [gid for gid, *_ in graph.scan()] == [0]
    assert len(list((graph @ graph).scan())) == 2
    assert graph @ graph == graph @ graph.saturate()


def test_Markov_rebuild():
    x = markov.Ty('x')
    graph = Markov.id(x @ x)
    a, b = graph.dom_wires
    f = markov.Box('f', x, x)
    (u, ), (v, ) = graph.intern(f, (a, )), graph.intern(f, (a, ))
    graph.rebuild()
    assert graph.uf.find(u) != graph.uf.find(v)


def test_Frobenius_laws():
    f_, g_, h_ = F(f), F(g), F(h)
    assert (f_ >> g_) >> h_ == f_ >> (g_ >> h_) == F(f >> g >> h)
    assert Frobenius.id(x) >> f_ == f_ == f_ >> Frobenius.id(y)
    assert f_ @ h_ == F(f @ Id(x) >> Id(y) @ h) == F(Id(x) @ h >> f @ Id(x))
    assert hash(f_ >> g_) == hash(F(f >> g))
    assert F(Swap(x, y) >> Swap(y, x)) == Frobenius.id(x @ y)
    with raises(AxiomError):
        f_ >> f_


def test_Traced_lowering():
    x_ = symmetric.Ty('x')
    G = Traced.from_diagram
    assert G(symmetric.Swap(x_, x_).trace()) == G(symmetric.Id(x_))
    loop = G(symmetric.Id(x_).trace())
    assert [box for _, box, *_ in loop.scan()] == [
        symmetric.Trace(symmetric.Id(x_))]


def test_Markov_lowering():
    x_ = markov.Ty('x')
    f_ = markov.Box('f', x_, x_)
    discarded = f_ >> markov.Discard(x_)
    assert Markov.from_diagram(discarded).to_diagram().to_hypergraph()\
        == discarded.to_hypergraph()
    assert len(Markov.from_diagram(markov.Copy(x_)).rows) == 1
    assert Cartesian.from_diagram(discarded).to_diagram()\
        == markov.Discard(x_)
    graph = Cartesian.from_diagram(markov.Copy(x_))
    assert not graph.rows and len(set(graph.uf)) == 1
    (u, ), (v, ) = graph.intern(f_, graph.cod_wires[:1]), graph.intern(
        f_, graph.cod_wires[1:])
    assert u == v


def test_Frobenius_fuse_spiders():
    graph = F(Spider(1, 2, x) >> Spider(2, 1, x))
    assert graph.fuse_spiders() == 1 and graph.fuse_spiders() == 0
    assert F(Spider(2, 1, x) >> Spider(1, 2, x)).equiv(F(Spider(2, 2, x)))
    phased = F(Spider(1, 1, x, .25) >> Spider(1, 1, x, .5))
    assert phased.equiv(F(Spider(1, 1, x, .75)))
    assert not phased.equiv(F(Spider(1, 1, x, .5)))
    assert not F(Spider(1, 2, x)).equiv(F(Spider(1, 1, x) @ Spider(0, 1, x)))


def test_Compact_yank_snakes():
    snake = Id(x) @ Cap(x.r, x) >> Cup(x, x.r) @ Id(x)
    assert F(snake).equiv(Frobenius.id(x))
    assert F(Cap(x, x.l) @ Id(x) >> Id(x) @ Cup(x.l, x)).equiv(
        Frobenius.id(x))
    assert F(Cap(x.r, x) >> Cup(x.r, x)).yank_snakes() == 0
    x_ = rigid.Ty('x')
    crossed = Compact[rigid.Diagram].id(x_.r)
    u, _ = crossed.intern(rigid.Cap(x_, x_.l), ())
    crossed.intern(rigid.Cup(x_, x_.r), (u, ) + crossed.dom_wires)
    assert crossed.yank_snakes() == 0


def test_Frobenius_alternatives():
    s, t = Box('s', Ty(), x), Box('t', Ty(), x)
    both = F(s)
    both.merge(*both.cod_wires, *both.intern(t, ()))
    assert both.equiv(F(s)) and both.equiv(F(t)) and not F(s).equiv(F(t))
    assert both.to_diagram() == s


def test_Frobenius_equiv():
    assert F(f).equiv(F(f)) and not F(f).equiv(F(Box('g', x, y)))
    assert not F(f).equiv(F(h))
    assert F(Swap(x, x) >> h @ h).equiv(F(h @ h >> Swap(x, x)))


def test_Frobenius_state_lift():
    graph = F(state @ state)
    graph.rebuild()
    assert len(list(graph.scan())) == 1
    assert len(graph.to_hypergraph().boxes) == 2


@mark.parametrize("diagram", [
    f, f >> g, f @ g, Id(x), Swap(x, y), Cap(x, x) >> Cup(x, x),
    Id(x) @ Cap(x, x) >> Cup(x, x) @ Id(x), Spider(2, 1, x),
    Spider(1, 1, x, .5), Spider(0, 0, x), state @ state, state >> f,
    f @ Id(x) >> Id(y) @ h, (state @ state) >> (h @ h)])
def test_round_trip(diagram):
    graph = F(diagram)
    assert graph.to_diagram().to_hypergraph() == diagram.to_hypergraph()


@mark.parametrize("factory, diagram", [
    (Traced, symmetric.Box(
        'f', symmetric.Ty('x', 'x'), symmetric.Ty('x', 'x')).trace()),
    (Traced, symmetric.Swap(symmetric.Ty('x'), symmetric.Ty('y'))),
    (Markov, markov.Copy(markov.Ty('x')) >> markov.Box(
        'f', markov.Ty('x', 'x'), markov.Ty('x')))])
def test_round_trip_levels(factory, diagram):
    graph = factory.from_diagram(diagram)
    assert graph.to_diagram().to_hypergraph() == diagram.to_hypergraph()
