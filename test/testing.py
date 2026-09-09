# -*- coding: utf-8 -*-

"""
One test for each argument generator of :mod:`discopy.testing`: it accepts
valid arguments, rejects invalid ones, and its search strategy reaches every
shape of argument the axioms expect. Whether the axioms hold is checked over
every category in ``proptest/``.
"""

from __future__ import annotations

from hypothesis import find
from pytest import raises

from discopy import biclosed, cat, feedback, monoidal, rigid, testing, traced
from discopy.testing import (
    C0, C1, Atomic, Axiom, AxiomFailure, Square, BoundaryConnected,
    ComposablePair, ComposableTriple, FeedbackJoining, FeedbackVanishing,
    HomogeneousMemory, HorizontalPair, LeftCurrying, Natural, NonEmpty,
    Relabelling, RightCurrying, Subsingleton, TraceDinaturalityLeft,
    TraceDinaturalityRight, TraceNaturalityLeft, TraceNaturalityRight,
    TraceSuperposing, axiom, resolve)
from discopy.utils import AxiomError


def test_Natural():
    assert Natural(2) @ Natural(3) == 5 == len(Natural(5))
    assert Natural(1).__matmul__("x") is NotImplemented
    assert repr(Natural(2)) == "testing.Natural(2)"
    assert eval(repr(Natural(2))) == testing.Natural(2)
    assert str(Natural(2)) == "2"
    assert Natural.equation_factory(Natural(1), Natural(1))
    with raises(ValueError):
        Natural(-1)
    find(Natural.strategy(), lambda value: value == 0)
    find(Natural.strategy(), lambda value: value > 1)


def test_Atomic():
    x, y = map(monoidal.Ty, "xy")
    assert Atomic(x).value == x
    with raises(ValueError):
        Atomic(x @ y)
    find(Atomic[monoidal.Ty].strategy(),
         lambda value: len(value.value) == 1)


def test_NonEmpty():
    x = monoidal.Ty('x')
    assert NonEmpty(x).value == x
    with raises(ValueError):
        NonEmpty(monoidal.Ty())
    find(NonEmpty[monoidal.Ty].strategy(),
         lambda value: len(value.value) > 1)
    nested = find(resolve(NonEmpty[ComposablePair[cat.Arrow]]), lambda _: True)
    assert isinstance(nested.value, ComposablePair)


def test_ComposablePair():
    x, y = map(cat.Ob, "xy")
    f, g = cat.Box('f', x, y), cat.Box('g', y, x)
    assert ComposablePair(f, g) == (f, g)
    with raises(ValueError):
        ComposablePair(f)
    with raises(AxiomError):
        ComposablePair(f, f)
    find(ComposablePair[cat.Arrow].strategy(),
         lambda value: all(term.inside for term in value))


def test_ComposableTriple():
    x, y = map(cat.Ob, "xy")
    f, g = cat.Box('f', x, y), cat.Box('g', y, x)
    assert ComposableTriple(f, g, f) == (f, g, f)
    with raises(AxiomError):
        ComposableTriple(f, f, f)
    find(ComposableTriple[cat.Arrow].strategy(),
         lambda value: all(term.inside for term in value))


def test_HorizontalPair():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert HorizontalPair(f, g) == (f, g)
    with raises(ValueError):
        HorizontalPair(f)
    find(HorizontalPair[monoidal.Diagram].strategy(),
         lambda value: all(term.boxes for term in value))


def test_Bifunctor():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert Square(f, f, g, g) == (f, f, g, g)
    with raises(AxiomError):
        Square(f, f, f, f)
    find(Square[monoidal.Diagram].strategy(),
         lambda value: all(
             value[column].boxes or value[column + 2].boxes
             for column in range(2)))


def test_TraceSuperposing():
    x, y, z = map(traced.Ty, "xyz")
    assert TraceSuperposing(traced.Id(x), y) == (traced.Id(x), y)
    with raises(AxiomError):
        TraceSuperposing(traced.Box('f', x, y), z)
    find(TraceSuperposing[traced.Diagram].strategy(),
         lambda value: len(value[1]) > 1)


def test_TraceNaturalityLeft():
    x, y = map(traced.Ty, "xy")
    f, g = traced.Box('f', x @ y, x @ x), traced.Box('g', x, y)
    assert TraceNaturalityLeft(f, x, g) == (f, x, g)
    with raises(ValueError):
        TraceNaturalityLeft(traced.Id(x @ y), x, traced.Id(x))
    find(TraceNaturalityLeft[traced.Diagram].strategy(),
         lambda value: value[2].dom != value[2].cod)


def test_TraceNaturalityRight():
    x, y = map(traced.Ty, "xy")
    f, g = traced.Box('f', y @ x, x @ x), traced.Box('g', x, y)
    assert TraceNaturalityRight(f, x, g) == (f, x, g)
    with raises(ValueError):
        TraceNaturalityRight(traced.Id(x @ y), x, traced.Id(y))
    find(TraceNaturalityRight[traced.Diagram].strategy(),
         lambda value: value[2].dom != value[2].cod)


def test_TraceDinaturalityLeft():
    x, y, z = map(traced.Ty, "xyz")
    f, g = traced.Box('f', x @ z, y @ z), traced.Box('g', y, x)
    assert TraceDinaturalityLeft(f, g) == (f, g)
    with raises(ValueError):
        TraceDinaturalityLeft(g, f)
    find(TraceDinaturalityLeft[traced.Diagram].strategy(),
         lambda value: value[1].dom != value[1].cod)


def test_TraceDinaturalityRight():
    x, y, z = map(traced.Ty, "xyz")
    f, g = traced.Box('f', z @ x, z @ y), traced.Box('g', y, x)
    assert TraceDinaturalityRight(f, g) == (f, g)
    with raises(ValueError):
        TraceDinaturalityRight(g, f)
    shape = find(TraceDinaturalityRight[traced.Diagram].strategy(),
                 lambda value: value[1].dom != value[1].cod)
    sliding = shape[1]
    assert shape[0].dom[-len(sliding.cod):] == sliding.cod
    assert shape[0].cod[-len(sliding.dom):] == sliding.dom


def test_LeftCurrying():
    x, y = map(biclosed.Ty, "xy")
    evaluation = biclosed.Diagram.ev(x, y, left=True)
    assert LeftCurrying(evaluation, x, y) == (evaluation, x, y)
    with raises(ValueError):
        LeftCurrying(evaluation, y, x)
    find(LeftCurrying[biclosed.Diagram].strategy(),
         lambda value: value[1] != value[2])


def test_RightCurrying():
    x, y = map(biclosed.Ty, "xy")
    evaluation = biclosed.Diagram.ev(x, y, left=False)
    assert RightCurrying(evaluation, x, y) == (evaluation, x, y)
    with raises(ValueError):
        RightCurrying(evaluation, y, x)
    find(RightCurrying[biclosed.Diagram].strategy(),
         lambda value: value[1] != value[2])


def test_FeedbackVanishing():
    x = feedback.Ty('x')
    f, unit = feedback.Box('f', x, x), feedback.Ty()
    assert FeedbackVanishing(f, unit) == (f, unit)
    with raises(ValueError):
        FeedbackVanishing(f, x)
    find(FeedbackVanishing[feedback.Diagram].strategy(),
         lambda value: value[0].boxes)


def test_FeedbackJoining():
    x, y, z = map(feedback.Ty, "xyz")
    memory = y @ z
    f = feedback.Box('f', x @ memory.delay(), x @ memory)
    assert FeedbackJoining(f, memory) == (f, memory)
    with raises(ValueError):
        FeedbackJoining(f, feedback.Ty())
    with raises(ValueError):
        FeedbackJoining(feedback.Box('g', x @ memory, x @ memory), memory)
    with raises(ValueError):
        FeedbackJoining(
            feedback.Box('g', x @ memory.delay(), x @ memory.delay()), memory)
    shape = find(FeedbackJoining[feedback.Diagram].strategy(),
                 lambda value: value[1][:1] != value[1][1:])
    assert shape[0].cod[-2:] == shape[1]


def test_HomogeneousMemory():
    x, m = map(feedback.Ty, "xm")
    f = feedback.Box('f', x @ (m @ m).delay(), x @ m @ m)
    assert HomogeneousMemory(f, m @ m)
    n = feedback.Ty('n')
    g = feedback.Box('g', x @ (m @ n).delay(), x @ m @ n)
    with raises(ValueError):
        HomogeneousMemory(g, m @ n)
    find(HomogeneousMemory[feedback.Diagram].strategy(),
         lambda value: True)


def test_Relabelling():
    x, y, z = cat.Ob('x'), cat.Ob('y'), cat.Ob('z')
    relabelling = Relabelling(((x, y), ))
    assert relabelling[x] == y and relabelling[z] == z
    assert list(relabelling) == [x] and len(relabelling) == 1
    assert bool(Relabelling())
    assert relabelling[cat.Box('f', x, x)] == cat.Box('f', y, y)
    rigid_x, rigid_y = rigid.Ty('x'), rigid.Ty('y')
    rotating = Relabelling(((rigid_x, rigid_y), ))
    functor = rigid.Functor(rotating, rotating)
    assert functor(rigid_x.l) == rigid_y.l and functor(rigid_x.r) == rigid_y.r
    rotated = rigid.Box('f', rigid_x.r, rigid_x @ rigid_x)
    assert rotating[rotated] == rigid.Box('f', rigid_y.r, rigid_y @ rigid_y)
    delayed = Relabelling(((feedback.Ty('u'), feedback.Ty('v')), ))
    delaying = feedback.Box('f', feedback.Ty('u').delay(), feedback.Ty('u'))
    assert delayed[delaying] == feedback.Box(
        'f', feedback.Ty('v').delay(), feedback.Ty('v'))


def test_Axiom():
    @axiom
    def law(cls, f):
        """ Not an equation. """
        return cls.equation_factory(f)

    assert repr(law) == "Axiom(law)"
    assert eval(repr(cat.Arrow.unitality)) == cat.Arrow.unitality
    assert hash(cat.Arrow.unitality) == hash(eval(repr(cat.Arrow.unitality)))
    assert cat.Arrow.unitality != cat.Functor.unitality
    assert cat.Functor.dagger_involution() is NotImplemented
    assert [parameter.name for parameter in law.parameters] == ['f']
    assert cat.Arrow.unitality.carrier is cat.Arrow
    with raises(TypeError):
        law(cat.Id(cat.Ob('x')))
    with raises(TypeError):
        law.falsify()
    with raises(TypeError):
        law.strategy()
    assert law.bind(cat.Arrow)(cat.Id(cat.Ob('x')))
    assert Axiom(classmethod(lambda cls: NotImplemented)).bind(cat.Arrow)()\
        is NotImplemented
    broken = cat.Arrow.unitality.weaken(f=Atomic[C1]).failing("Never holds.")
    assert broken.subspaces == {"f": Atomic[C1]}
    with raises(AxiomFailure) as failure:
        broken(Atomic(cat.Box('f', cat.Ob('x'), cat.Ob('y'))))
    assert failure.value.equation


def test_modulo():
    law = cat.Arrow.unitality.modulo(lambda term: term.dom)
    assert law(cat.Box('f', cat.Ob('x'), cat.Ob('y')))


def test_weaken():
    for subspace in (Atomic[C1], Atomic[monoidal.Ty]):
        law = monoidal.Ty.monoid_unitality.weaken(x=subspace)
        assert law.modulo(lambda term: term).subspaces == law.subspaces
        args = find(law.strategy(), lambda _: True)
        assert isinstance(args[0], Atomic) and law(*args)


def test_element_law():
    @axiom
    def preserves_identity(self, x: C0) -> cat.Equation:
        """ A functor preserves the identity on each object. """
        return cat.Equation(self(cat.Arrow.id(x)), cat.Arrow.id(self(x)))

    law = preserves_identity.bind(cat.Functor)
    assert law.is_method
    args = find(law.strategy(), lambda _: True)
    assert law(*args)


def test_inapplicable():
    class Carrier(cat.Arrow):
        unitality = cat.Arrow.unitality.inapplicable("No identities.")

    unitality = Carrier.axioms["unitality"]
    assert unitality() is NotImplemented
    assert unitality.__doc__ == "No identities."
    assert not unitality.parameters and not unitality.broken
    dropped = cat.Arrow.unitality.weaken(f=Atomic[C1])\
        .inapplicable("No identities.")
    assert dropped.name == "unitality" and not dropped.subspaces


def test_Small():
    x = monoidal.Ty('x')
    assert Subsingleton(x).value == x
    with raises(ValueError):
        Subsingleton(x @ x)
    find(Subsingleton[monoidal.Ty].strategy(),
         lambda value: len(value.value) == 1)
    with raises(TypeError):
        resolve(int)


def test_BoundaryConnected():
    x = monoidal.Ty('x')
    f = monoidal.Box('f', x, x)
    scalar = monoidal.Box('s', monoidal.Ty(), monoidal.Ty())
    assert BoundaryConnected(f).value == f
    assert BoundaryConnected(f.to_hypergraph()).value
    assert BoundaryConnected(HorizontalPair(f, f)).value == (f, f)
    for value in (
            f @ scalar, scalar, scalar.to_map(), scalar.to_hypergraph()):
        with raises(ValueError):
            BoundaryConnected(value)
    find(BoundaryConnected[monoidal.Diagram].strategy(),
         lambda value: bool(value.value.boxes))
