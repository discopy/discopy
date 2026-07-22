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


def test_Box_hash_invariant_under_hypergraph_equality():
    """
    A box is a generator, so its hash must not depend on whether we use
    hypergraph equality, see https://github.com/discopy/discopy/issues/382
    """
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    outside = hash(f)
    with Diagram.hypergraph_equality:
        inside = hash(f)
    assert outside == inside == hash(f)
    # A box stored in a dictionary must still be found inside the context.
    boxes = {f: 42}
    with Diagram.hypergraph_equality:
        assert boxes[f] == 42
        assert boxes[Box('f', x, y)] == 42


def test_Functor_hypergraph_equality():
    """
    Regression test for https://github.com/discopy/discopy/issues/382

    A ``Functor`` stores the image of each generator by hashing the boxes in
    its domain.  Turning on ``hypergraph_equality`` used to change the hash of
    a box, so the box went missing from the functor's dictionary and its
    evaluation raised a ``KeyError``.
    """
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    F = Functor(ob_map={x: y, y: x}, ar_map={f: g, g: f})
    assert F(f) == g and F(g) == f
    with Diagram.hypergraph_equality:
        assert F(f) == g and F(g) == f
        assert F(f >> g) == g >> f


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
