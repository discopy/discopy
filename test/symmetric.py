# -*- coding: utf-8 -*-

from pytest import raises

from discopy.symmetric import *
from discopy.utils import AxiomError


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
    assert perm == Diagram((Layer(perm),), perm.dom, perm.cod)
    assert Permutation.id(x @ y @ z) == Id(x @ y @ z)
    assert Permutation.id(x @ y @ z).inside == ()
    assert perm >> perm.dagger() == Id(x @ y @ z)
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
    assert Permutation(x @ y, [1, 0]) == Swap(x, y)
    assert hash(Permutation(x @ y, [1, 0])) == hash(Swap(x, y))
    with raises(ValueError):
        Permutation(x @ y @ z, [0, 1, 2])
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
    perm_layer = Layer(p)
    assert perm_layer.boxes == [] and perm_layer.dom == x @ y
    assert perm_layer.cod == y @ x
    assert layer.dagger().dom == layer.cod and layer.dagger().cod == layer.dom
    assert layer.dagger().dagger() == layer
    assert (z @ layer).dom == z @ layer.dom
    assert (layer @ z).dom == layer.dom @ z
    assert Layer.cast(f) == Layer(Ty(), f, Ty())
    assert Layer.cast(p) == Layer(p)
    with raises(ValueError):
        Layer(x, f)
    with raises(ValueError):
        Layer(x, p, y)


def test_permutation_factory():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    functor = Functor(ob={x: y, y: z, z: x}, ar={})
    assert functor(perm) == Permutation(y @ z @ x, [2, 0, 1])
    with Diagram.hypergraph_equality:
        assert perm == perm.to_swaps()
        assert functor(perm) == functor(perm.to_swaps())


def test_Permutation_whiskering():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation(x @ y, [1, 0])
    assert perm @ z == Permutation(x @ y @ z, [1, 0, 2]) == perm @ Id(z)
    assert z @ perm == Permutation(z @ x @ y, [0, 2, 1])
    assert isinstance(perm @ z, Permutation)
    assert isinstance(z @ perm, Permutation)
    assert perm @ Ty() == perm == Ty() @ perm
    assert perm.tensor() == perm == perm.then()


def test_Permutation_foliation():
    x, y, z, w = map(Ty, "xyzw")
    f0, f1 = Box("f0", w, x), Box("f1", z, y)
    g0, g1 = Box("g0", y, z), Box("g1", x, w)
    reverse = Permutation(x @ y @ z @ w, [3, 2, 1, 0])
    diagram = (reverse >> f0 @ f1 @ g0 @ g1).foliation()
    assert diagram.inside == (Layer(reverse), Layer(
        Ty(), f0, Ty(), f1, Ty(), g0, Ty(), g1, Ty()))
    with raises(AxiomError):
        diagram.inside[0].merge(diagram.inside[1])


def test_Functor_on_composite_types():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm = Permutation(x @ y, [1, 0])
    functor = Functor(ob={x: y @ z, y: z}, ar={})
    assert functor(perm) == Diagram.swap(y @ z, z)
