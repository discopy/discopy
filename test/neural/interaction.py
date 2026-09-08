# -*- coding: utf-8 -*-

from pytest import raises

from discopy import neural, symmetric
from discopy.interaction import Int
from discopy.neural import network
from discopy.neural.interaction import (
    Box, CMap, Cap, Cup, Diagram, Dim, Dims, Equation, Functor, Id, Leg,
    Network, Swap, Ty)
from discopy.utils import AxiomError, dumps, loads


def test_leg():
    x = Leg(Dim(2, 3))
    assert x == Leg(Dim(2, 3)) != Leg(Dim(6)) and x.dim == Dim(2, 3)
    assert x.r == x.l == x.dagger() == Leg(Dim(2, 3), z=1) and x.r.r == x
    assert x.r.unwind() == x and str(x.r) == "Dim(2, 3).r"
    assert repr(x.r) == "Leg(Dim(2, 3), z=1)" and repr(Leg(2)) == "Leg(Dim(2))"
    assert loads(dumps(x.r)) == x.r and hash(x) == hash(Leg(Dim(2, 3)))
    with raises(TypeError):
        Leg(Dims(2))


def test_ty():
    x = Ty(2, Dim(2, 3))
    assert x == Ty(Dims(2, Dim(2, 3))) == Ty(Leg(2), Leg(Dim(2, 3)))
    assert x.l == x.r == Ty(Leg(Dim(2, 3), z=1), Leg(2, z=1)) and x.r.r == x
    assert str(x @ x.r) == "Dim(2) @ Dim(2, 3) @ Dim(2, 3).r @ Dim(2).r"
    assert repr(x.r) == "Ty(Leg(Dim(2, 3), z=1), Leg(Dim(2), z=1))"
    assert (x @ x.r).positive == (x @ x.r).negative == Dims(2, Dim(2, 3))
    assert (x.r @ x).negative == x.positive == (x.r @ x).positive
    assert x.to_int() == Int(Network).ob(Dims(2, Dim(2, 3)))
    assert (x @ x.r).to_int() == x.to_int() @ -x.to_int()
    assert Ty().to_int() == Int(Network).ob()
    assert loads(dumps(x @ x.r)) == x @ x.r and hash(x) == hash(Ty(x.positive))
    with raises(TypeError):
        Ty('x')


def test_axioms():
    x = Ty(2)
    assert Equation(Id(x.r).transpose(left=True), Id(x), Id(x.r).transpose())
    assert Equation(Cap(x, x.r) >> Swap(x, x.r), Cap(x.r, x))
    assert Equation(Swap(x, x.r) >> Cup(x.r, x), Cup(x, x.r))
    with raises(AxiomError):
        Cup(x, x)


def test_box():
    k = network.Box('k', Dims(2, 5), Dims(3, 4))
    K = Box('k', Ty(2, Leg(4, z=1)), Ty(3, Leg(5, z=1)), network=k)
    assert K.network is K.data is k and K.to_network() == k
    assert K.r.r == K and K.r.network == K.dagger().network == k
    assert K.r.dom == Ty(5, Leg(3, z=1)) and K.r.cod == Ty(4, Leg(2, z=1))
    assert K.r.to_network() == Network.swap(Dims(5), Dims(2)) >> k\
        >> Network.swap(Dims(3), Dims(4))
    assert K.dagger().to_network() == k.dagger()
    assert K.r.dagger().to_network() == K.dagger().r.to_network()
    assert Box('k', K.dom, K.cod, data=k) == K != Box('k', K.dom, K.cod)
    for box in (K, K.r, K.dagger(), K.r.dagger()):
        assert loads(dumps(box)) == box
    scope = {"neural": neural, "symmetric": symmetric, "Ty": Ty, "Leg": Leg,
             "Dim": Dim, "Dims": Dims}
    for box in (K, K.r, K.dagger(), Box('f', Ty(2), Ty(3)).r):
        assert eval(repr(box), scope) == box
    with raises(ValueError):
        Box('k', K.cod, K.dom, network=k)
    with raises(ValueError):
        Box('k', K.dom, Ty(3), network=k)
    with raises(ValueError):
        Box('k', K.dom, Ty(3, Leg(5, z=1), 7), network=k)
    with raises(TypeError):
        Box('k', K.dom, K.cod, network=K)
    with raises(ValueError):
        Box('f', Ty(2), Ty(3)).to_network()


def test_execution_formula():
    x, y = Ty(2), Ty(3)
    f = network.Box('f', Dims(2), Dims(3))
    g = network.Box('g', Dims(3), Dims(1))
    F, G = Box('f', x, y, network=f), Box('g', y, Ty(1), network=g)
    execution = (F >> G).to_map().to_network()
    assert execution.dom == Dims(2) and execution.cod == Dims(1)
    assert network.Equation(execution, f >> g)
    assert network.Equation((F @ G).to_map().to_network(), f @ g)
    assert F.to_map().to_network()\
        == (network.Swap(Dims(2), Dims(3)) >> Dims(3) @ f).trace()
    assert F.transpose().to_map().to_network() == F.to_map().to_network()
    assert network.Equation(F.transpose().to_int().inside, f)
    assert network.Equation(F.dagger().to_int().inside, f.dagger())
    assert network.Equation((F >> G).dagger().to_int().inside, (f >> g)[::-1])


def test_to_int_is_a_functor():
    x, y = Ty(2), Ty(3)
    f = network.Box('f', Dims(2), Dims(3))
    g = network.Box('g', Dims(3), Dims(1))
    F, G = Box('f', x, y, network=f), Box('g', y, Ty(1), network=g)
    composed, tensored = (F >> G).to_int(), (F @ G.r).to_int()
    assert composed.dom == x.to_int() and composed.cod == Ty(1).to_int()
    assert network.Equation(composed.inside, (F.to_int() >> G.to_int()).inside)
    assert network.Equation(tensored.inside, (F.to_int() @ G.r.to_int()).inside)
    assert network.Equation(Id(x @ y.r).to_int().inside, Network.id(Dims(2, 3)))
    assert network.Equation(
        Swap(x, y.r).to_int().inside,
        Int(Network).braid(x.to_int(), y.r.to_int()).inside)
    assert network.Equation(
        Diagram.permutation([2, 0, 1], [x, y, x.r]).to_int().inside,
        Int(Network).permutation(
            [2, 0, 1], [x.to_int(), y.to_int(), x.r.to_int()]).inside)
    for snake in (Id(x).transpose(), Id(x.r).transpose(left=True)):
        assert network.Equation(snake.to_int().inside, Network.id(Dims(2)))


def test_backward_legs():
    x, h = Dims(2), Dims(4)
    f = network.Box('f', x, h)
    k = network.Box('k', x @ x, h @ h)
    G = Box('g', Ty(2, Leg(4, z=1)), Ty(), network=f)
    K = Box('k', Ty(2, Leg(4, z=1)), Ty(4, Leg(2, z=1)), network=k)
    assert G.to_int().dom == Ty(2, Leg(4, z=1)).to_int()
    assert network.Equation(G.to_int().inside, f)
    assert network.Equation(G.r.to_int().inside, G.r.to_network())
    assert network.Equation(K.r.to_int().inside, K.r.to_network())
    assert network.Equation(K.dagger().to_int().inside, k.dagger())
    assert network.Equation(
        (K @ G.r).to_int().inside, (K.to_int() @ G.r.to_int()).inside)


def test_recurrent():
    x, h = Dims(2), Dims(4)
    cell = network.Box('cell', x @ h, x @ h)
    C = Box('cell', Ty(x @ h), Ty(x @ h), network=cell)
    recurrent = (C >> C).trace()
    assert not recurrent.to_map().is_acyclic
    assert network.Equation(
        recurrent.to_map().to_network(), (cell >> cell).trace())
    assert network.Equation(
        recurrent.trace(left=True).to_int().inside,
        (cell >> cell).trace().trace(left=True))
    loop = (Cap(Ty(2), Ty(2).r) >> Cup(Ty(2), Ty(2).r)).to_map()
    assert loop.loops == (Ty(2), )
    assert loop.to_network() == Network.id(Dims(2)).trace()


def test_cmap():
    x, y = Ty(2), Ty(3)
    f = network.Box('f', Dims(2), Dims(3))
    F = Box('f', x, y, network=f)
    assert isinstance(F.to_map(), CMap) and isinstance(F.to_map() >> F.dagger().to_map(), CMap)
    assert CMap.from_diagram(F) == F.to_map() == CMap.from_box(F)
    assert F.to_map().to_diagram() == F
    assert F.transpose().to_map().boxes == (F, )
    assert F.to_map().to_int().inside == F.to_map().to_network()
    with raises(ValueError):
        Box('f', x, y).to_map().to_network()


def test_functor():
    x, y, z = Ty(2), Ty(3), Ty(1)
    f = network.Box('f', Dims(2), Dims(3))
    g = network.Box('g', Dims(3), Dims(1))
    F, G = Box('f', x, y, network=f), Box('g', y, z, network=g)
    H = Functor(ob_map={x: z, y: y}, ar_map={F: G.dagger()})
    assert H(F >> F.dagger()) == G.dagger() >> G
    assert H(F.r) == G.dagger().r and H(F.transpose()) == G.dagger().transpose()
    assert H(x @ y.r) == z @ y.r and H(Cup(x, x.r)) == Cup(z, z.r)
    assert H(Swap(x, y)) == Swap(z, y)
