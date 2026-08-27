# -*- coding: utf-8 -*-

"""
One test for each argument generator of :mod:`discopy.testing`: it accepts
valid arguments, rejects invalid ones, and its search strategy reaches every
shape of argument the axioms expect. Whether the axioms hold is checked over
every category in ``proptest/``.
"""

from hypothesis import find
from pytest import raises

from discopy import biclosed, cat, feedback, monoidal, traced
from discopy.testing import (
    Atomic, Bifunctor, BoundaryConnected, ComposablePair, ComposableTriple,
    FeedbackJoining, FeedbackVanishing, HomogeneousMemory, HorizontalPair,
    LeftCurrying, Natural, NonEmpty, RightCurrying, Small,
    TraceDinaturalityLeft, TraceDinaturalityRight, TraceNaturalityLeft,
    TraceNaturalityRight, TraceSuperposing, axiom)
from discopy.utils import AxiomError


def test_Natural():
    assert Natural(2) @ Natural(3) == 5 == len(Natural(5))
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
    find(Atomic.strategy(factory=monoidal.Ty),
         lambda value: len(value.value) == 1)


def test_NonEmpty():
    x = monoidal.Ty('x')
    assert NonEmpty(x).value == x
    with raises(ValueError):
        NonEmpty(monoidal.Ty())
    find(NonEmpty.strategy(factory=monoidal.Ty),
         lambda value: len(value.value) > 1)


def test_ComposablePair():
    x, y = map(cat.Ob, "xy")
    f, g = cat.Box('f', x, y), cat.Box('g', y, x)
    assert ComposablePair(f, g) == (f, g)
    with raises(ValueError):
        ComposablePair(f)
    with raises(AxiomError):
        ComposablePair(f, f)
    find(ComposablePair.strategy(factory=cat.Arrow),
         lambda value: all(term.inside for term in value))


def test_ComposableTriple():
    x, y = map(cat.Ob, "xy")
    f, g = cat.Box('f', x, y), cat.Box('g', y, x)
    assert ComposableTriple(f, g, f) == (f, g, f)
    with raises(AxiomError):
        ComposableTriple(f, f, f)
    find(ComposableTriple.strategy(factory=cat.Arrow),
         lambda value: all(term.inside for term in value))


def test_HorizontalPair():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert HorizontalPair(f, g) == (f, g)
    with raises(ValueError):
        HorizontalPair(f)
    find(HorizontalPair.strategy(factory=monoidal.Diagram),
         lambda value: all(term.boxes for term in value))


def test_Bifunctor():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert Bifunctor(f, f, g, g) == (f, f, g, g)
    with raises(AxiomError):
        Bifunctor(f, f, f, f)
    find(Bifunctor.strategy(factory=monoidal.Diagram),
         lambda value: all(
             value[column].boxes or value[column + 2].boxes
             for column in range(2)))


def test_TraceSuperposing():
    x, y, z = map(traced.Ty, "xyz")
    assert TraceSuperposing(traced.Id(x), y) == (traced.Id(x), y)
    with raises(AxiomError):
        TraceSuperposing(traced.Box('f', x, y), z)
    find(TraceSuperposing.strategy(factory=traced.Diagram),
         lambda value: len(value[1]) > 1)


def test_TraceNaturalityLeft():
    x, y = map(traced.Ty, "xy")
    f, g = traced.Box('f', x @ y, x @ x), traced.Box('g', x, y)
    assert TraceNaturalityLeft(f, x, g) == (f, x, g)
    with raises(ValueError):
        TraceNaturalityLeft(traced.Id(x @ y), x, traced.Id(x))
    find(TraceNaturalityLeft.strategy(factory=traced.Diagram),
         lambda value: value[2].dom != value[2].cod)


def test_TraceNaturalityRight():
    x, y = map(traced.Ty, "xy")
    f, g = traced.Box('f', y @ x, x @ x), traced.Box('g', x, y)
    assert TraceNaturalityRight(f, x, g) == (f, x, g)
    with raises(ValueError):
        TraceNaturalityRight(traced.Id(x @ y), x, traced.Id(y))
    find(TraceNaturalityRight.strategy(factory=traced.Diagram),
         lambda value: value[2].dom != value[2].cod)


def test_TraceDinaturalityLeft():
    x, y, z = map(traced.Ty, "xyz")
    f, g = traced.Box('f', x @ z, y @ z), traced.Box('g', y, x)
    assert TraceDinaturalityLeft(f, g) == (f, g)
    with raises(ValueError):
        TraceDinaturalityLeft(g, f)
    find(TraceDinaturalityLeft.strategy(factory=traced.Diagram),
         lambda value: value[1].dom != value[1].cod)


def test_TraceDinaturalityRight():
    x, y, z = map(traced.Ty, "xyz")
    f, g = traced.Box('f', z @ x, z @ y), traced.Box('g', y, x)
    assert TraceDinaturalityRight(f, g) == (f, g)
    with raises(ValueError):
        TraceDinaturalityRight(g, f)
    shape = find(TraceDinaturalityRight.strategy(factory=traced.Diagram),
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
    find(LeftCurrying.strategy(factory=biclosed.Diagram),
         lambda value: value[1] != value[2])


def test_RightCurrying():
    x, y = map(biclosed.Ty, "xy")
    evaluation = biclosed.Diagram.ev(x, y, left=False)
    assert RightCurrying(evaluation, x, y) == (evaluation, x, y)
    with raises(ValueError):
        RightCurrying(evaluation, y, x)
    find(RightCurrying.strategy(factory=biclosed.Diagram),
         lambda value: value[1] != value[2])


def test_FeedbackVanishing():
    x = feedback.Ty('x')
    f, unit = feedback.Box('f', x, x), feedback.Ty()
    assert FeedbackVanishing(f, unit) == (f, unit)
    with raises(ValueError):
        FeedbackVanishing(f, x)
    find(FeedbackVanishing.strategy(factory=feedback.Diagram),
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
    shape = find(FeedbackJoining.strategy(factory=feedback.Diagram),
                 lambda value: value[1][:1] != value[1][1:])
    assert shape[0].cod[-2:] == shape[1]


def test_Axiom():
    @axiom
    def law(cls, f):
        """ Not an equation. """
        return cls.equation_factory(f)

    assert repr(law) == "Axiom(law)"
    assert [parameter.name for parameter in law.parameters] == ['f']
    assert cat.Arrow.unitality.carrier is cat.Arrow
    with raises(TypeError):
        law(cat.Id(cat.Ob('x')))
    assert law.bind(cat.Arrow)(cat.Id(cat.Ob('x')))


def test_inapplicable():
    class Carrier(cat.Arrow):
        unitality = cat.Arrow.unitality.inapplicable("No identities.")

    unitality, = (a for a in Carrier.axioms if a.name == "unitality")
    assert unitality() is NotImplemented
    assert unitality.__doc__ == "No identities."
    assert not unitality.parameters and not unitality.broken


def test_Small():
    x = monoidal.Ty('x')
    assert Small(x).value == x
    with raises(ValueError):
        Small(x @ x)
    find(Small.strategy(factory=monoidal.Ty),
         lambda value: len(value.value) == 1)


def test_BoundaryConnected():
    x = monoidal.Ty('x')
    f = monoidal.Box('f', x, x)
    scalar = monoidal.Box('s', monoidal.Ty(), monoidal.Ty())
    assert BoundaryConnected(f).value == f
    with raises(ValueError):
        BoundaryConnected(f @ scalar)
    find(BoundaryConnected.strategy(factory=monoidal.Diagram),
         lambda value: bool(value.value.boxes))


def test_HomogeneousMemory():
    x, m = map(feedback.Ty, "xm")
    f = feedback.Box('f', x @ (m @ m).delay(), x @ m @ m)
    assert HomogeneousMemory(f, m @ m)
    n = feedback.Ty('n')
    g = feedback.Box('g', x @ (m @ n).delay(), x @ m @ n)
    with raises(ValueError):
        HomogeneousMemory(g, m @ n)
    find(HomogeneousMemory.strategy(factory=feedback.Diagram),
         lambda value: True)


def test_weaken():
    from discopy.matrix import Matrix

    weakened, = (axiom for axiom in Matrix[int].axioms
                 if axiom.name == "copy_cocommutativity_small")
    assert weakened.subspaces
    assert not weakened.broken
    small = find(weakened.strategy(), lambda args: True)
    assert isinstance(small[0], Small) and weakened(*small)
