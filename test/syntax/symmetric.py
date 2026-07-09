# -*- coding: utf-8 -*-

from pytest import raises

from discopy.symmetric import *


def test_Swap():
    x, y = Ty('x'), Ty('y')
    assert repr(Swap(x, y))\
        == "symmetric.Swap(monoidal.Ty(cat.Ob('x')), monoidal.Ty(cat.Ob('y')))"
    assert Swap(x, y).dagger() == Swap(y, x)
    with raises(ValueError):
        Swap(x ** 2, Ty())


def test_Box_hash():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    assert f == f @ Id()
    assert hash(f) == hash(f @ Id())
    assert hash(f) == hash(Id() @ f)
    assert f @ Id() in {f}
    assert {f: 42}[f @ Id()] == 42


def test_Box_hash_hypergraph():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    with Diagram.hypergraph_equality:
        assert f == f @ Id()
        assert hash(f) == hash(f @ Id())
        assert f @ Id() in {f}


def test_Functor_hypergraph_equality():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    F = Functor(ob={x: x, y: y}, ar={f: g[::-1], g: f[::-1]})
    with Diagram.hypergraph_equality:
        assert F(f >> g) == g[::-1] >> f[::-1]
    with Diagram.hypergraph_equality:
        G = Functor(ob={x: x, y: y}, ar={f: g[::-1], g: f[::-1]})
    assert G(f >> g) == g[::-1] >> f[::-1]


def test_Diagram_permutation():
    x = PRO(1)
    tmp, Diagram.ob = Diagram.ob, PRO
    assert Diagram.swap(x, x ** 2)\
        == Diagram.swap(x, x) @ Id(x) >> Id(x) @ Diagram.swap(x, x)\
        == Diagram.permutation([1, 2, 0])\
        == Diagram.permutation([2, 0, 1]).dagger()
    with raises(ValueError):
        Diagram.permutation([2, 0])
    with raises(ValueError):
        Diagram.permutation([2, 0, 1], x ** 2)
    Diagram.ob = tmp


def test_bad_permute():
    with raises(ValueError):
        Id(Ty('n')).permute(1)
    with raises(ValueError):
        Id(Ty('n')).permute(0, 0)
