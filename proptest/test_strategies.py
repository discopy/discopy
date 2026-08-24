"""Tests for the Hypothesis strategy layer."""

import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st

from discopy import (
    balanced,
    biclosed,
    braided,
    cat,
    closed,
    compact,
    feedback,
    frobenius,
    markov,
    matrix,
    monoidal,
    pivotal,
    ribbon,
    rigid,
    symmetric,
    testing,
    traced,
)
from discopy.utils import AxiomError
from discopy.python import finset

from proptest import strategies


def test_shapes_validate_at_construction():
    x, y = map(cat.Ob, "xy")
    f, g = cat.Box("f", x, y), cat.Box("g", y, x)
    assert testing.ComposablePair(f, g) == (f, g)
    with pytest.raises(AxiomError):
        testing.ComposablePair(f, f)
    with pytest.raises(ValueError):
        testing.Atomic(monoidal.Ty())
    with pytest.raises(ValueError):
        testing.NonEmpty(monoidal.Ty())


@pytest.mark.parametrize("factory", (
    monoidal.Diagram, compact.Diagram,
    monoidal.Hypergraph, compact.CMap,
))
@given(data=st.data())
@settings(max_examples=25)
def test_generated_boxes_are_distinct(factory, data):
    value = data.draw(strategies.strategy(factory))
    assert len(set(value.boxes)) == len(value.boxes)


def test_axiom_annotations_are_resolved_in_the_bound_scope():
    axiom = monoidal.Diagram.associativity
    assert strategies.arguments(axiom) is not None
    assert axiom.carrier is monoidal.Diagram


def test_recursive_category_uses_its_box_strategy():
    arrow = find(
        strategies.strategy(cat.Arrow, max_leaves=3),
        lambda value: bool(value.inside))
    assert all(isinstance(box, cat.Box) for box in arrow.inside)


def test_min_leaves_reaches_strategy_pipeline(monkeypatch):
    calls, recursive = [], st.recursive

    def record(*args, **params):
        calls.append((params["min_leaves"], params["max_leaves"]))
        return recursive(*args, **params)

    monkeypatch.setattr(st, "recursive", record)
    params = dict(min_leaves=2, max_leaves=3)
    for factory in (cat.Arrow, finset.Function):
        factory.strategy(**params)
    assert calls == [(2, 3)] * 2
    for factory in (
            monoidal.Diagram, monoidal.Hypergraph, monoidal.CMap):
        diagram = find(factory.strategy(**params), lambda value: True)
        assert len(diagram.boxes) >= 2


def test_diagrams_are_generated_layer_by_layer():
    x = monoidal.Ty("x")
    composition = find(monoidal.Diagram.strategy(
        types=st.just(x), min_leaves=2, max_leaves=2),
        lambda value: True)
    assert len(composition.boxes) == 2
    assert len(composition.inside) == 2
    assert all(
        isinstance(layer, monoidal.Layer)
        for layer in composition.inside)
    assert all(
        isinstance(box, monoidal.Box) for box in composition.boxes)


@pytest.mark.parametrize("Diagram", (
    monoidal.Diagram, symmetric.Diagram, compact.Diagram,
))
def test_diagrams_are_boundary_connected_by_default(Diagram):
    diagram = find(Diagram.strategy(), lambda value: True)
    assert diagram.to_hypergraph().is_boundary_connected


def test_diagrams_can_generate_closed_components():
    diagram = find(monoidal.Diagram.strategy(boundary_connected=False),
                   lambda value: not value.to_hypergraph()
                   .is_boundary_connected)
    assert not diagram.to_hypergraph().is_boundary_connected


def test_cmaps_can_generate_closed_components():
    cmap = find(compact.CMap.strategy(boundary_connected=False),
                lambda value: not value.to_hypergraph()
                .is_boundary_connected)
    assert not cmap.to_hypergraph().is_boundary_connected


def test_layers_own_boundary_guided_generation():
    x = monoidal.Ty("x")
    layer = find(monoidal.Layer.strategy(
        factory=monoidal.Diagram, dom=x), lambda value: True)
    assert layer.dom == x


def test_symmetric_layers_generate_simultaneous_permutations():
    dom = symmetric.Ty(*"xyz")
    layer = find(symmetric.Layer.strategy(
        factory=symmetric.Diagram, dom=dom),
        lambda value: value.is_plumbing)
    permutation, = layer.boxes
    assert isinstance(permutation, symmetric.Permutation)
    assert layer.dom == dom


def test_strategy_follows_the_category_hierarchy():
    assert issubclass(testing.ComposableTriple, testing.Strategy)
    assert issubclass(testing.ComposableTriple, testing.PastingDiagram)
    assert issubclass(monoidal.Diagram, testing.Strategy)
    assert issubclass(compact.Box, braided.Box)
    assert issubclass(compact.Box, rigid.Box)
    assert compact.Diagram.strategy.__func__\
        is monoidal.Diagram.strategy.__func__

    x = compact.Ty("x")
    box = compact.Box("f", x, x)
    assert box.generator is box


@pytest.mark.parametrize("module", (
    monoidal, braided, traced, balanced, symmetric, markov,
    biclosed, closed, rigid, pivotal, ribbon, compact, feedback,
    frobenius,
))
def test_every_diagram_level_inherits_its_box_factory(module):
    assert module.Diagram.box_factory is module.Box


@pytest.mark.parametrize("structure", (
    compact.Swap,
    compact.Cup,
    compact.Cap,
    ))
def test_compact_diagrams_generate_structural_morphisms(structure):
    box = find(compact.Box.strategy(),
               lambda value: isinstance(value, structure))
    assert isinstance(box, compact.Diagram)


@pytest.mark.parametrize("structure", (
    compact.Swap,
    compact.Cup,
    compact.Cap,
    ))
def test_diagrams_generate_structural_morphisms(structure):
    diagram = find(compact.Diagram.strategy(),
                   lambda value: any(isinstance(box, structure)
                                     for box in value.boxes))
    assert any(isinstance(box, structure) for box in diagram.boxes)


def test_braids_are_generated_in_both_orientations():
    for is_dagger in (False, True):
        box = find(braided.Box.strategy(),
                   lambda value: isinstance(value, braided.Braid)
                   and value.is_dagger == is_dagger)
        assert box.is_dagger == is_dagger


def test_matrices_are_generated_with_exact_boundaries():
    generated = find(matrix.Matrix.strategy(dom=2, cod=3),
                     lambda value: bool(value.array.any()))
    assert (generated.dom, generated.cod) == (2, 3)
    assert generated.array.shape == (2, 3)


def test_trace_dinaturality_slides_between_distinct_objects():
    shape = find(
        strategies.strategy(
            testing.TraceDinaturalityRight[compact.Diagram]),
        lambda value: value[1].dom != value[1].cod)
    traced, sliding = shape
    assert traced.dom[-len(sliding.cod):] == sliding.cod
    assert traced.cod[-len(sliding.dom):] == sliding.dom


def test_feedback_joining_generates_heterogeneous_memory():
    shape = find(
        strategies.strategy(testing.FeedbackJoining[feedback.Diagram]),
        lambda value: value[1][:1] != value[1][1:])
    arrow, memory = shape
    assert len(memory) == 2
    assert arrow.cod[-2:] == memory


@pytest.mark.parametrize(
    "shape", (testing.ComposablePair, testing.ComposableTriple))
def test_composable_chains_generate_every_term(shape):
    chain = find(
        strategies.strategy(shape[cat.Arrow]),
        lambda value: all(term.inside for term in value))
    assert all(
        left.is_composable(right)
        for left, right in zip(chain, chain[1:]))


def test_all_type_lookup_goes_through_the_trait():
    assert strategies.strategy(monoidal.Diagram) is not None
    with pytest.raises(TypeError):
        strategies.strategy(int)
