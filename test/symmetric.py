# -*- coding: utf-8 -*-

from pytest import raises

from discopy.symmetric import *
from discopy import monoidal


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
    assert perm.inside == (monoidal.Layer(Ty(), perm, Ty()),)
    assert perm.boxes == [perm] and perm.offsets == [0] and perm.size == 1
    assert Diagram.decode(*perm.encode()) == perm
    assert Permutation.id(x @ y @ z) == Id(x @ y @ z)
    assert Permutation.id(x @ y @ z).inside == ()
    assert perm >> perm.dagger() != Id(x @ y @ z)
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    assert perm.dagger().dagger() == perm
    a = Permutation(x @ y @ z, [1, 2, 0])
    b = Permutation(a.cod, [2, 0, 1])
    c = Permutation(b.cod, [0, 2, 1])
    assert (a >> b) >> c == a >> (b >> c)
    assert Id(x @ y @ z) >> a == a == a >> Id(a.cod)
    assert (a >> b).dagger() == b.dagger() >> a.dagger()
    q = Permutation(z @ y, [1, 0])
    assert Equation((perm @ q).dagger(), perm.dagger() @ q.dagger())
    assert (perm @ q).dom == perm.dom @ q.dom
    assert Permutation(x @ y, [1, 0]) != Swap(x, y)
    assert Equation(Permutation(x @ y, [1, 0]), Swap(x, y))
    with raises(ValueError):
        Permutation(x @ y @ z, [0, 1, 2])
    with raises(ValueError):
        Permutation(x @ y @ z, [2, 0])
    with raises(ValueError):
        Permutation(x @ y @ z, [0, 0, 1])


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
        assert other @ q == p @ q
        assert other.dagger() == p.dagger()
    assert (p >> q) >> f == p >> (q >> f)
    assert (p @ q) @ f == p @ (q @ f)


def test_permutation_factory():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    functor = Functor(ob_map={x: y, y: z, z: x}, ar_map={})
    assert functor(perm) == Permutation(y @ z @ x, [2, 0, 1])
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
    assert Equation(perm @ z, Permutation(x @ y @ z, [1, 0, 2]))
    assert Equation(z @ perm, Permutation(z @ x @ y, [0, 2, 1]))
    assert perm @ z == perm @ Id(z)
    assert perm @ Ty() == perm == Ty() @ perm
    assert perm.tensor() == perm == perm.then()


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
        monoidal.Layer(Ty(), reverse, Ty()), monoidal.Layer(
            Ty(), f0, Ty(), f1, Ty(), g0, Ty(), g1, Ty()))
    assert foliated.boxes == [reverse, f0, f1, g0, g1]


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


def test_Functor_on_composite_types():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    perm, other = (
        Permutation(x @ y, [1, 0]), Permutation(y @ x, [1, 0]))
    functor = Functor(ob_map={x: y @ z, y: z}, ar_map={})
    assert functor(perm) == Permutation(y @ z @ z, [2, 0, 1])
    assert functor(perm >> other) == functor(perm) >> functor(other)
    assert functor(perm @ other) == functor(perm) @ functor(other)
    assert functor(perm.dagger()) == functor(perm).dagger()

    erase = Functor(ob_map={x: Ty(), y: z}, ar_map={})
    assert erase(perm) == Id(z)
