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


def test_Permutation():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation(x @ y @ z, [1, 2, 0])
    assert perm.dom == x @ y @ z and perm.cod == y @ z @ x
    assert list(perm.perm) == [1, 2, 0]
    # a permutation box is equal to the diagram with just that permutation
    assert perm == Diagram((Layer(perm),), perm.dom, perm.cod)
    # composition with the inverse is the identity diagram
    assert perm >> perm.dagger() == Permutation.id(x @ y @ z)
    assert perm.dagger().dagger() == perm
    # composition is associative and respects identities
    a = Permutation(x @ y @ z, [1, 2, 0])
    b = Permutation(a.cod, [2, 0, 1])
    c = Permutation(b.cod, [0, 2, 1])
    assert (a >> b) >> c == a >> (b >> c)
    assert Permutation.id(x @ y @ z) >> a == a == a >> Permutation.id(a.cod)
    assert (a >> b).dagger() == b.dagger() >> a.dagger()
    # tensor is functorial
    q = Permutation(z @ y, [1, 0])
    assert (perm @ q).dagger() == perm.dagger() @ q.dagger()
    assert (perm @ q).dom == perm.dom @ q.dom
    # the permutation (1, 0) is the swap, even though stored differently
    assert Permutation(x @ y, [1, 0]) == Swap(x, y)
    assert hash(Permutation(x @ y, [1, 0])) == hash(Swap(x, y))
    # the identity permutation is the identity diagram, not a Permutation box
    with raises(ValueError):
        Permutation(x @ y @ z, [0, 1, 2])
    # wrong permutations are rejected
    with raises(ValueError):
        Permutation(x @ y @ z, [2, 0])
    with raises(ValueError):
        Permutation(x @ y @ z, [0, 0, 1])


def test_Layer():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    p = Permutation(x @ y, [1, 0])
    layer = Layer(p, f, y)
    assert layer.dom == x @ y @ x @ y and layer.cod == y @ x @ y @ y
    assert layer.boxes == [f]
    # a permutation-only layer has a single argument and no box
    perm_layer = Layer(p)
    assert perm_layer.boxes == [] and perm_layer.dom == x @ y
    assert perm_layer.cod == y @ x
    # dagger is component-wise (parallel) and involutive
    assert layer.dagger().dom == layer.cod and layer.dagger().cod == layer.dom
    assert layer.dagger().dagger() == layer
    # whiskering with a type extends the boundary permutations
    assert (z @ layer).dom == z @ layer.dom
    assert (layer @ z).dom == layer.dom @ z
    # cast turns a box into a layer, a permutation into a one-argument layer
    assert Layer.cast(f) == Layer(Ty(), f, Ty())
    assert Layer.cast(p) == Layer(p)
    # layers must have an odd number of elements
    with raises(ValueError):
        Layer(x, f)
    # permutations may not sit at odd indices
    with raises(ValueError):
        Layer(x, p, y)


def test_permutation_factory():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    # a symmetric functor maps a permutation to a permutation of the image
    functor = Functor(ob={x: y, y: z, z: x}, ar={})
    assert functor(perm) == Permutation(y @ z @ x, [2, 0, 1])
    # and it agrees with the same permutation expanded into swaps
    with Diagram.hypergraph_equality:
        assert perm == perm.to_swaps()
        assert functor(perm) == functor(perm.to_swaps())
