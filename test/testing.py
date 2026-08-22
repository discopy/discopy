# -*- coding: utf-8 -*-

"""
Tests for the property-test data structures of :mod:`discopy.testing`.

The strategies themselves are checked against every category in ``proptest/``,
here we check the validation of arguments and the binding of axioms.
"""

from hypothesis import find, given, settings
from hypothesis import strategies as st
import pytest
from pytest import raises

from discopy import (
    balanced, biclosed, braided, cat, feedback, frobenius, markov, monoidal,
    rigid, symmetric, traced)
from discopy.python import finset
from discopy.testing import (
    Atomic, Bifunctor, ComposablePair, ComposableTriple, FeedbackJoining,
    FeedbackVanishing, HorizontalPair, LeftCurrying, Natural, NonEmpty,
    RightCurrying, TraceNaturalityLeft, TraceNaturalityRight,
    TraceSuperposing, axiom)
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
    x, y = map(traced.Ty, "xy")
    with raises(ValueError):
        TraceNaturalityLeft(traced.Id(x @ y), x, traced.Id(x))
    with raises(ValueError):
        TraceNaturalityLeft(traced.Id(y @ x), x, traced.Id(y))


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
    extended = balanced.Box.extend_strategy(
        base, balanced.Diagram.twist_factory, build)
    assert type(find(extended, lambda box: type(box) is balanced.Box))\
        is balanced.Box
    assert isinstance(find(
        extended, lambda box: isinstance(box, balanced.Twist)),
        balanced.Twist)


def test_arrow_strategy_with_boundaries_is_recursive():
    x, y = map(cat.Ob, "xy")
    arrow = find(cat.Arrow.strategy(
        dom=x, cod=y, min_leaves=2, max_leaves=2),
        lambda value: len(value.inside) > 1)
    assert arrow.dom == x and arrow.cod == y


def test_layer_strategy_excludes_boxes():
    x = monoidal.Ty("x")
    params = dict(
        factory=monoidal.Diagram, types=st.just(x), dom=x, cod=x,
        label=0)
    first = find(monoidal.Layer.strategy(**params), lambda _: True)
    second = find(monoidal.Layer.strategy(
        **params, exclude=first.boxes), lambda _: True)
    assert not set(first.boxes).intersection(second.boxes)


def test_unconstrained_layer_strategy():
    layer = find(monoidal.Layer.strategy(
        factory=monoidal.Diagram), lambda _: True)
    assert isinstance(layer, monoidal.Layer)


@pytest.mark.parametrize(("shape", "factory"), (
    (Atomic, monoidal.Ty),
    (NonEmpty, monoidal.Ty),
    (ComposablePair, cat.Arrow),
    (ComposableTriple, cat.Arrow),
    (HorizontalPair, monoidal.Diagram),
    (Bifunctor, monoidal.Diagram),
    (TraceSuperposing, traced.Diagram),
    (TraceNaturalityLeft, traced.Diagram),
    (TraceNaturalityRight, traced.Diagram),
    (LeftCurrying, biclosed.Diagram),
    (RightCurrying, biclosed.Diagram),
    (FeedbackVanishing, feedback.Diagram),
    (FeedbackJoining, feedback.Diagram),
))
def test_argument_strategy(shape, factory):
    assert find(shape.strategy(factory=factory), lambda _: True) is not None


@pytest.mark.parametrize(("factory", "structure"), (
    (braided.Box, braided.Braid),
    (traced.Box, traced.Trace),
    (biclosed.Box, biclosed.Eval),
    (rigid.Box, rigid.Cup),
    (markov.Box, markov.Copy),
    (feedback.Box, feedback.Feedback),
    (frobenius.Box, frobenius.Spider),
))
def test_box_strategy_generates_structure(factory, structure):
    value = find(factory.strategy(), lambda box: isinstance(box, structure))
    assert isinstance(value, structure)


def test_diagram_strategy_generates_closed_components():
    diagram = find(monoidal.Diagram.strategy(boundary_connected=False),
                   lambda value: not value.to_hypergraph()
                   .is_boundary_connected)
    assert not diagram.to_hypergraph().is_boundary_connected


def test_symmetric_layer_strategy_from_codomain_and_types():
    cod = symmetric.Ty(*"xyz")
    from_cod = find(symmetric.Layer.strategy(
        factory=symmetric.Diagram, cod=cod),
        lambda layer: layer.is_plumbing)
    assert from_cod.cod == cod
    unconstrained = find(symmetric.Layer.strategy(
        factory=symmetric.Diagram, types=st.just(cod)),
        lambda layer: layer.is_plumbing)
    assert unconstrained.is_plumbing


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
