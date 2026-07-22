# -*- coding: utf-8 -*-

from pytest import raises

from discopy.symmetric import *
from discopy.utils import factory


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


def test_swap_factory():
    """
    Setting ``swap_factory`` on a custom subclass sets ``braid_factory``,
    so its swaps stay in the subclass all the way through ``Diagram.swap``,
    ``Hypergraph.to_diagram`` and ``Diagram.from_callable``,
    see https://github.com/discopy/discopy/issues/395
    """
    @factory
    class Ingredient(Ty):
        pass

    @factory
    class Recipe(Diagram):
        ob = Ingredient

    class CookingStep(Box, Recipe):
        pass

    class CookingSwap(Swap, CookingStep):
        pass

    Recipe.swap_factory = CookingSwap
    assert Recipe.swap_factory is Recipe.braid_factory is CookingSwap
    white, yolk = Ingredient("white"), Ingredient("yolk")
    swap = Recipe.swap(white, yolk)
    assert isinstance(swap, CookingSwap)
    roundtrip = swap.to_hypergraph().to_diagram()
    assert isinstance(roundtrip, Recipe)
    assert all(isinstance(box, CookingSwap) for box in roundtrip.boxes)
    diagram = Recipe.from_callable(white @ yolk, yolk @ white)(
        lambda white, yolk: (yolk, white))
    assert diagram == swap

    class Permutation(Diagram):
        pass

    class Transposition(Swap, Permutation):
        pass

    class SubPermutation(Permutation):
        swap_factory = Transposition

    assert SubPermutation.braid_factory is Transposition
