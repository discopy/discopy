# -*- coding: utf-8 -*-

import pickle

from pytest import raises

from discopy.symmetric import *
from discopy.utils import AxiomError, dumps, loads


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


def test_symmetric_Equation():
    """
    ``symmetric.Equation`` compares diagrams up to hypergraph isomorphism
    (e.g. swaps cancel) while ``==`` stays syntactic,
    see https://github.com/discopy/discopy/issues/382
    """
    x = Ty('x')
    a, b = Swap(x, x) >> Swap(x, x), Id(x @ x)
    assert a != b
    assert Equation(a, b)
    assert not Equation(a, Swap(x, x))


def test_Box_hash_is_syntactic_and_stable():
    """
    Equality and hashing are always syntactic, so a box and the length-one
    diagram made of it are equal and hash equally, and a box can always be
    found in a dict, see https://github.com/discopy/discopy/issues/382
    """
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    assert f == f @ Id() and hash(f) == hash(f @ Id())
    assert f @ Id() in {f}
    assert {f: 42}[Box('f', x, y)] == 42


def test_Functor_keys_boxes_by_syntax():
    """
    Regression test for https://github.com/discopy/discopy/issues/382

    A ``Functor`` stores the image of each generator by hashing the boxes in
    its domain.  Now that equality and hashing are always syntactic, box
    lookups are stable and functor application never loses a box.
    """
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    F = Functor(ob_map={x: y, y: x}, ar_map={f: g, g: f})
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


def test_Permutation():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation(x @ y @ z, [1, 2, 0])
    assert perm.dom == x @ y @ z and perm.cod == y @ z @ x
    assert list(perm.perm) == [1, 2, 0]
    assert perm.inside == (Layer(perm),)
    assert perm.boxes == [] and perm.size == 0
    assert not perm.is_generator and perm.generator is None
    with raises(NotImplementedError):
        perm.encode()
    identity = Permutation(x @ y @ z, [0, 1, 2])
    assert identity.is_identity and identity == Id(x @ y @ z)
    assert identity.inside == () and identity.boxes == []
    assert Permutation.id(x @ y @ z) == Id(x @ y @ z)
    assert Permutation.id(x @ y @ z).inside == ()
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    assert perm.dagger().dagger() == perm
    a = Permutation(x @ y @ z, [1, 2, 0])
    b = Permutation(a.cod, [2, 0, 1])
    c = Permutation(b.cod, [0, 2, 1])
    assert (a >> b) >> c == a >> (b >> c)
    assert Id(x @ y @ z) >> a == a == a >> Id(a.cod)
    assert (a >> b).dagger() == b.dagger() >> a.dagger()
    q = Permutation(z @ y, [1, 0])
    assert (perm @ q).dagger() == perm.dagger() @ q.dagger()
    assert (perm @ q).dom == perm.dom @ q.dom
    assert Permutation(x @ y, [1, 0]) != Swap(x, y)
    assert Equation(Permutation(x @ y, [1, 0]), Swap(x, y))
    with raises(ValueError):
        Permutation(x @ y @ z, [2, 0])
    with raises(ValueError):
        Permutation(x @ y @ z, [0, 0, 1])


def test_Layer():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    swap, permutation = Swap(x, y), Permutation(x @ y, [1, 0])
    layer = Layer(x, f, y)
    assert all(
        isinstance(component, Permutation) for component in layer[::2])
    assert all(component.is_identity for component in layer[::2])
    assert layer.boxes_and_types == (x, f, y)
    assert layer.boxes == [f]
    assert Layer.cast(f) == Layer(Ty(), f, Ty())
    assert Layer.cast(permutation) == Layer(permutation)
    assert Layer(x, swap, z).boxes == [swap]
    assert permutation != swap
    assert layer.dagger().dagger() == layer
    assert (z @ layer).boxes_and_types == (z @ x, f, y)
    assert (layer @ z).boxes_and_types == (x, f, y @ z)
    assert all(
        isinstance(component, Permutation)
        for component in (z @ layer)[::2] + (layer @ z)[::2])
    with raises(ValueError):
        Layer(x, f)
    with raises(ValueError):
        Layer(x, permutation, y)


def test_Layer_serialisation():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    permutation = Permutation(x @ y, [1, 0])
    for layer in (Layer(x, f, y), Layer(permutation)):
        result = loads(dumps(layer))
        assert result == layer
        assert all(
            isinstance(component, Permutation) for component in result[::2])
        result = pickle.loads(pickle.dumps(layer))
        assert result == layer
        assert all(
            isinstance(component, Permutation) for component in result[::2])
    identity = Permutation(x @ y, [0, 1])
    assert isinstance(loads(dumps(identity)), Permutation)


def test_Layer_identity():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    identity = Layer(x)
    layer = Layer.cast(f)
    assert identity.boxes_and_offsets == []
    assert identity.merge(layer) == layer
    assert layer.merge(Layer(y)) == layer


def test_Layer_factory_ownership():
    from discopy import compact, markov

    for module in (compact, markov):
        x, y = module.Ty('x'), module.Ty('y')
        permutation = module.Permutation(x @ y, [1, 0])
        layer = module.Layer(permutation)
        assert type(layer.permutation) is module.Permutation
        f = module.Box('f', x, y)
        layer = module.Layer(x[:0], f, y[:0])
        assert all(type(p) is module.Permutation for p in layer[::2])


def test_Layer_tensor():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    left, right = Layer(x, f, y), Layer(z, g, x)
    result = left @ right
    assert result.boxes_and_types == (x, f, y @ z, g, x)
    assert result.boxes == [f, g]
    assert all(
        isinstance(component, Permutation) for component in result[::2])
    assert (left @ right).dagger() == left.dagger() @ right.dagger()
    permutation = Layer(Permutation(x @ y, [1, 0]))
    assert (left @ right) @ permutation == left @ (right @ permutation)


def test_noncommuting_Permutation_composition():
    x = Ty('x')
    first = Permutation(x ** 3, [1, 0, 2])
    second = Permutation(first.cod, [0, 2, 1])
    composite = first >> second
    assert Equation(composite, Permutation(x ** 3, [1, 2, 0]))
    assert not Equation(composite, Permutation(x ** 3, [2, 0, 1]))


def test_Permutation_box_setoid():
    x = Ty("x")
    p = Permutation(x ** 3, [1, 2, 0])
    q = p.dagger()
    f = Box("f", x ** 3, x ** 3)
    representatives = p, p[:], Id(p.dom) >> p, p >> Id(p.cod)
    for other in representatives:
        assert other == p and hash(other) == hash(p)
        assert other >> q == p >> q
        assert q >> other == q >> p
        assert Equation(other @ q, p @ q)
        assert other.dagger() == p.dagger()
    assert (p >> q) >> f == p >> (q >> f)
    assert Equation((p @ q) @ f, p @ (q @ f))


def test_permutation_factory():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    functor = Functor(ob_map={x: y, y: z, z: x}, ar_map={})
    assert Equation(
        functor(perm), Permutation(y @ z @ x, [2, 0, 1]))
    assert Equation(perm, perm.to_swaps())
    assert Equation(functor(perm), functor(perm.to_swaps()))


def test_inherited_permutation_factory():
    from discopy import closed, feedback, frobenius, tensor

    cases = [
        (closed, closed.Ty("x"), closed.Ty("y")),
        (feedback, feedback.Ty("x"), feedback.Ty("y")),
        (frobenius, frobenius.Ty("x"), frobenius.Ty("y")),
        (tensor, tensor.Dim(2), tensor.Dim(3))]
    for module, x, y in cases:
        perm = module.Diagram.from_permutation([1, 0], x @ y)
        assert isinstance(perm, module.Diagram)
        assert perm.ar is module.Diagram
        assert not any(
            isinstance(box, Permutation) for box in perm.boxes)


def test_Permutation_whiskering():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation(x @ y, [1, 0])
    assert perm @ z == Permutation(x @ y @ z, [1, 0, 2])
    assert z @ perm == Permutation(z @ x @ y, [0, 2, 1])
    assert isinstance(perm @ z, Permutation)
    assert isinstance(z @ perm, Permutation)
    assert perm @ z == perm @ Id(z)
    assert perm @ Ty() == perm == Ty() @ perm
    assert perm.tensor() == perm == perm.then()
    identity = Permutation(x, [0])
    assert isinstance(identity @ y, Permutation)
    assert isinstance(y @ identity, Permutation)


def test_mixed_Layer_routing():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    permutation = Permutation(x @ y, [1, 0])
    f = Box('f', z, z)
    layer = Layer(permutation, f, Ty())
    diagram = Diagram((layer,), layer.dom, layer.cod)
    assert diagram.dagger().dagger() == diagram
    assert diagram.to_hypergraph().dom == diagram.dom
    assert diagram.to_drawing().dom == diagram.dom.to_drawing()
    assert loads(dumps(diagram)) == diagram
    assert pickle.loads(pickle.dumps(diagram)) == diagram
    with raises(NotImplementedError):
        layer.boxes_and_types
    with raises(NotImplementedError):
        diagram.normalize()
    with raises(NotImplementedError):
        diagram.interchange(0, 0)
    with raises(NotImplementedError):
        diagram.substitute(0, f)


def test_Permutation_foliation():
    x, y, z, w = map(Ty, "xyzw")
    f0, f1 = Box("f0", w, x), Box("f1", z, y)
    g0, g1 = Box("g0", y, z), Box("g1", x, w)
    reverse = Permutation(x @ y @ z @ w, [3, 2, 1, 0])
    diagram = reverse >> f0 @ f1 @ g0 @ g1
    foliated = diagram.foliation()
    assert Equation(diagram, foliated)
    assert foliated.depth() == 1
    assert foliated.inside == (
        Layer(reverse), Layer(
            Ty(), f0, Ty(), f1, Ty(), g0, Ty(), g1, Ty()))
    assert foliated.boxes == [f0, f1, g0, g1]
    with raises(AxiomError):
        foliated.inside[0].merge(foliated.inside[1])


def test_large_Permutation_to_hypergraph():
    x, n = Ty("x"), 1100
    perm = Permutation(x ** n, reversed(range(n)))
    graph = perm.to_hypergraph()
    assert graph.boxes == ()
    assert graph.cod_wires == tuple(reversed(range(n)))


def test_default_Permutation_to_hypergraph():
    perm = Diagram.from_permutation([2, 1, 0])
    graph = perm.to_hypergraph()
    assert graph.cod == PRO(3)
    assert graph.cod_wires == (2, 1, 0)
    assert Equation(perm, perm.to_swaps())


def test_Permutation_to_drawing():
    from discopy.drawing import Drawing

    x, y, z = map(Ty, 'xyz')
    perm = Permutation(x @ y, [1, 0])
    functor = Functor(
        ob_map={x: z, y: x}, ar_map={}, cod=Drawing)
    drawing = functor(perm)
    assert drawing.dom == (z @ x).to_drawing()
    assert drawing.cod == (x @ z).to_drawing()
