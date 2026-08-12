# -*- coding: utf-8 -*-

"""
Tests for the property-test data structures of :mod:`discopy.testing`.

The strategies themselves are checked against every category in ``proptest/``,
here we check the validation of arguments and the binding of axioms.
"""

from hypothesis import given, settings
from pytest import raises

from discopy import balanced, cat, feedback, monoidal, traced
from discopy.python import finset
from discopy.testing import (
    Atomic, ComposablePair, FeedbackJoining, FeedbackVanishing, Natural,
    NonEmpty, TraceSliding, axiom)
from discopy.utils import AxiomError


def test_Natural():
    assert Natural(2) @ Natural(3) == 5 == len(Natural(5))
    with raises(ValueError):
        Natural(-1)


def test_Atomic():
    x, y = map(monoidal.Ty, "xy")
    assert Atomic(x).value == x
    with raises(ValueError):
        Atomic(x @ y)


def test_NonEmpty():
    x = monoidal.Ty('x')
    assert NonEmpty(x).value == x
    with raises(ValueError):
        NonEmpty(monoidal.Ty())


def test_PastingDiagram():
    x, y = map(monoidal.Ty, "xy")
    f = monoidal.Box('f', x, y)
    with raises(ValueError):
        ComposablePair(f)
    with raises(AxiomError):
        ComposablePair(f, f)


def test_TraceSliding():
    x = traced.Ty('x')
    with raises(ValueError):
        TraceSliding(traced.Id(x @ x), x, traced.Id(x @ x))
    with raises(ValueError):
        TraceSliding(traced.Id(x), x, traced.Id(x))


def test_Feedback():
    x = feedback.Ty('x')
    f = feedback.Box('f', x, x)
    with raises(ValueError):
        FeedbackVanishing(f, x)
    with raises(ValueError):
        FeedbackJoining(f, feedback.Ty())


def test_Axiom():
    @axiom(strict=False)
    def law(cls, f, *, eq):
        """ Not an equation. """
        return eq(f)

    assert repr(law) == "Axiom(law)" and law.strict is False
    assert [parameter.name for parameter in law.parameters] == ['f']
    assert cat.Arrow.unitality.carrier is cat.Arrow
    with raises(TypeError):
        law(cat.Id(cat.Ob('x')))
    assert law.bind(cat.Arrow)(cat.Id(cat.Ob('x')))


def test_extend_strategy():
    base = balanced.Box.free_strategy()
    build = lambda factory: balanced.Box.atomic_strategy().map(factory)
    assert balanced.Box.extend_strategy(
        base, balanced.Diagram.twist_factory, build,
        dom=balanced.Ty('x')) is base


@given(box=balanced.Box.strategy(label="f"))
@settings(max_examples=5, deadline=None)
def test_box_strategy(box):
    assert balanced.Id(box.dom) >> box == box >> balanced.Id(box.cod)


@given(function=finset.Function.strategy())
@settings(max_examples=5, deadline=None)
def test_finset_strategy(function):
    assert function.then(finset.Function.id(function.cod)) == function


@given(function=finset.Function.generator_strategy(cod=1, max_size=1))
@settings(max_examples=20, deadline=None)
def test_finset_generator_strategy(function):
    """ There is no function from the empty set to a non-empty one. """
    assert function.cod == 1 and function.dom == 1
