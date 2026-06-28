# -*- coding: utf-8 -*-

import random

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
    tmp, Diagram.ty_factory = Diagram.ty_factory, PRO
    assert Diagram.swap(x, x ** 2)\
        == Diagram.swap(x, x) @ Id(x) >> Id(x) @ Diagram.swap(x, x)\
        == Diagram.permutation([1, 2, 0])\
        == Diagram.permutation([2, 0, 1]).dagger()
    with raises(ValueError):
        Diagram.permutation([2, 0])
    with raises(ValueError):
        Diagram.permutation([2, 0, 1], x ** 2)
    Diagram.ty_factory = tmp


def test_bad_permute():
    with raises(ValueError):
        Id(Ty('n')).permute(1)
    with raises(ValueError):
        Id(Ty('n')).permute(0, 0)


def test_Permutation():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation([1, 2, 0], x @ y @ z)
    assert perm.cod == y @ z @ x
    assert perm.dom == x @ y @ z
    # composition with the inverse is the identity
    assert perm.then(perm.dagger()) == Permutation.id(x @ y @ z)
    assert perm.dagger().dagger() == perm
    # composition is associative and respects identities
    a = Permutation([1, 2, 0], PRO(3))
    b = Permutation([2, 0, 1], PRO(3))
    c = Permutation([0, 2, 1], PRO(3))
    assert (a >> b) >> c == a >> (b >> c)
    assert Permutation.id(PRO(3)) >> a == a == a >> Permutation.id(PRO(3))
    assert (a >> b).dagger() == b.dagger() >> a.dagger()
    # tensor is functorial
    q = Permutation([1, 0], z @ y)
    assert (perm @ q).dagger() == perm.dagger() @ q.dagger()
    assert (perm @ q).dom == perm.dom @ q.dom
    # wrong permutations are rejected
    with raises(ValueError):
        Permutation([2, 0], x @ y @ z)
    with raises(ValueError):
        Permutation([0, 0, 1], x @ y @ z)


def test_Layer():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', z, x)
    layer = Layer(Permutation([1, 0], x @ y), f, Permutation.id(y))
    assert layer.dom == x @ y @ x @ y
    assert layer.cod == y @ x @ y @ y
    assert layer.boxes == (f,)
    # a permutation-only layer has length one
    perm_layer = Layer(Permutation([1, 0], x @ y))
    assert perm_layer.boxes == () and perm_layer.dom == x @ y
    assert perm_layer.cod == y @ x
    # tensor merges the boundary permutations
    other = Layer(Permutation.id(z), g, Permutation.id(x))
    combined = layer @ other
    assert combined.dom == layer.dom @ other.dom
    assert combined.cod == layer.cod @ other.cod
    assert len(combined.boxes) == 2
    # dagger is component-wise (parallel) and involutive
    assert layer.dagger().dom == layer.cod
    assert layer.dagger().cod == layer.dom
    assert layer.dagger().dagger() == layer
    assert (layer @ other).dagger() == layer.dagger() @ other.dagger()
    # whiskering with a permutation absorbs into the boundary
    sigma = Permutation([1, 0], z @ x)
    assert (sigma @ layer).dom == sigma.dom @ layer.dom
    assert (layer @ sigma).dom == layer.dom @ sigma.dom
    # layers must have an odd number of elements
    with raises(ValueError):
        Layer(Permutation.id(x), f)


def _check_foliation(d, depth=None):
    layers = d.to_layers()
    # consecutive layers compose and match the diagram boundary
    for first, second in zip(layers, layers[1:]):
        assert first.cod == second.dom
    assert layers[0].dom == d.dom
    assert layers[-1].cod == d.cod
    # foliating then rebuilding gives back the same diagram
    with Diagram.hypergraph_equality:
        assert Diagram.from_layers(layers) == d
    n_box_layers = sum(1 for layer in layers if layer.boxes)
    if depth is not None:
        assert n_box_layers == depth
    return layers


def test_to_layers():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    # sequential boxes give one box-layer each, parallel boxes share a layer
    _check_foliation(f >> g, depth=2)
    _check_foliation(f @ Box('h', z, x), depth=1)
    # identities and pure permutations are a single permutation-only layer
    assert len(Id(x @ y).to_layers()) == 1
    assert not Id(x @ y).to_layers()[0].boxes
    _check_foliation(Swap(x, y), depth=0)
    _check_foliation(Diagram.permutation([2, 0, 1], x @ y @ z), depth=0)
    # boxes with unequal arity, with and without interleaved swaps
    split, merge = Box('split', x, y @ z), Box('merge', y @ z, x)
    _check_foliation(split >> merge, depth=2)
    _check_foliation(split @ x >> y @ z @ split, depth=1)
    _check_foliation(split >> Swap(y, z) >> z @ y @ Id(Ty()), depth=1)
    # a deeper diagram with a permutation rearranging the middle
    p = Box('p', x, y @ z)
    q, r = Box('q', y, x), Box('r', z, x)
    s = Box('s', x @ x, x)
    _check_foliation(p >> q @ r >> s, depth=3)
    _check_foliation(p >> Swap(y, z) >> r @ q >> s, depth=3)


def test_to_layers_random():
    random.seed(1234)
    types = [Ty('x'), Ty('y'), Ty('z')]

    def random_type(min_len, max_len=2):
        n = random.randint(min_len, max_len)
        return Ty().tensor(*(random.choice(types) for _ in range(n)))

    def random_diagram(width, n_ops, allow_empty):
        diagram = Id(Ty().tensor(*(random.choice(types) for _ in range(width))))
        for _ in range(n_ops):
            cod = diagram.cod
            if len(cod) == 0:
                break
            if random.random() < 0.4 and len(cod) >= 2:
                i = random.randint(0, len(cod) - 2)
                diagram >>= cod[:i] @ Swap(
                    cod[i:i + 1], cod[i + 1:i + 2]) @ cod[i + 2:]
            else:
                i = random.randint(0, len(cod) - 1)
                p = random.randint(1, min(2, len(cod) - i))
                box = Box("b", cod[i:i + p], random_type(0 if allow_empty else 1))
                diagram >>= cod[:i] @ box @ cod[i + p:]
        return diagram

    for allow_empty in (False, True):
        for _ in range(200):
            diagram = random_diagram(
                random.randint(1, 4), random.randint(0, 8), allow_empty)
            # depth() is only reliable when every box has non-empty dom and cod
            reliable = all(len(b.dom) and len(b.cod) for b in diagram.boxes)
            _check_foliation(
                diagram, depth=diagram.depth() if reliable else None)
