# -*- coding: utf-8 -*-

"""
Tests for :mod:`discopy.abc`, i.e. that the axioms declared by each abstract
category can be instantiated in the free category that implements it.

Whether the equations hold is checked over every category in ``proptest/``.
"""

from inspect import signature

import pytest

from discopy import (
    abc, balanced, biclosed, braided, cat, closed, compact, feedback,
    frobenius, markov, monoidal, pivotal, ribbon, rigid, symmetric, traced,
    utils)
from discopy.utils import AxiomError
from discopy.testing import (
    Atomic, ComposablePair, ComposableTriple, HorizontalPair, NonEmpty,
    Relabelled, Relabelling, assert_verdict, declared_axioms)


def box(category, name, dom, cod):
    factory = getattr(category, "box_factory", None) or category.generator_factory
    return factory(name, dom, cod)


class Arguments:
    """Canonical well-typed arguments for each categorical axiom."""

    @staticmethod
    def unitality(category):
        if issubclass(category, abc.ColouredMonoid):
            return category("f"),
        x, y = map(category.ob, "xy")
        return box(category, "f", x, y),

    @staticmethod
    def associativity(category):
        if issubclass(category, abc.ColouredMonoid):
            return tuple(map(category, "fgh")),
        x, y, z, w = map(category.ob, "xyzw")
        return (box(category, "f", x, y), box(category, "g", y, z),
                box(category, "h", z, w)),

    @staticmethod
    def identity_typing(category):
        return category.ob("x"),

    @staticmethod
    def composition_dom_typing(category):
        if issubclass(category, abc.ColouredMonoid):
            return tuple(map(category, "fg")),
        x, y, z = map(category.ob, "xyz")
        return (box(category, "f", x, y), box(category, "g", y, z)),

    @staticmethod
    def composition_cod_typing(category):
        if issubclass(category, abc.ColouredMonoid):
            return tuple(map(category, "fg")),
        x, y, z = map(category.ob, "xyz")
        return (box(category, "f", x, y), box(category, "g", y, z)),

    @staticmethod
    def monoid_unitality(category):
        return category("x"),

    @staticmethod
    def monoid_associativity(category):
        return tuple(map(category, "xyz")),

    @staticmethod
    def bifunctoriality(category):
        w, x, y, z, a, b = map(category.ob, "wxyzab")
        return (box(category, "f", w, x), box(category, "g", y, z),
                box(category, "h", x, a), box(category, "k", z, b)),

    @staticmethod
    def tensor_unitality(category):
        x, y, z, w = map(category.ob, "xyzw")
        return (box(category, "f", x, y), box(category, "g", z, w)),

    @staticmethod
    def tensor_dom_typing(category):
        x, y, z, w = map(category.ob, "xyzw")
        return (box(category, "f", x, y), box(category, "g", z, w)),

    @staticmethod
    def tensor_cod_typing(category):
        x, y, z, w = map(category.ob, "xyzw")
        return (box(category, "f", x, y), box(category, "g", z, w)),

    @staticmethod
    def trace_vanishing(category):
        x, y = map(category.ob, "xy")
        return box(category, "f", x, y),

    @staticmethod
    def trace_superposing_left(category):
        x, y, z = map(category.ob, "xyz")
        return (box(category, "f", x @ y, x @ z), x),

    @staticmethod
    def trace_superposing_right(category):
        x, y, z = map(category.ob, "xyz")
        return (box(category, "f", y @ x, z @ x), x),

    @staticmethod
    def trace_naturality_left(category):
        x, y = map(category.ob, "xy")
        return (box(category, "f", x @ y, x @ x), x,
                box(category, "g", x, y)),

    @staticmethod
    def trace_naturality_right(category):
        x, y = map(category.ob, "xy")
        return (box(category, "f", y @ x, x @ x), x,
                box(category, "g", x, y)),

    @staticmethod
    def trace_dinaturality_left(category):
        x, y, z = map(category.ob, "xyz")
        return (box(category, "f", x @ z, y @ z),
                box(category, "g", y, x)),

    @staticmethod
    def trace_dinaturality_right(category):
        x, y, z = map(category.ob, "xyz")
        return (box(category, "f", z @ x, z @ y),
                box(category, "g", y, x)),

    @staticmethod
    def currying_left(category):
        x, y = map(category.ob, "xy")
        return (box(category, "f", (x << y) @ y, x), x, y),

    @staticmethod
    def currying_right(category):
        x, y = map(category.ob, "xy")
        return (box(category, "f", y @ (y >> x), x), x, y),

    @staticmethod
    def adjunction(category):
        return category("x"),

    @staticmethod
    def snake_equations(category):
        return category.ob("x"),

    @staticmethod
    def caps_coherence(category):
        x, y = map(category.ob, "xy")
        return NonEmpty(x), NonEmpty(y)

    @staticmethod
    def self_dual(category):
        return category.ob("x"),

    @staticmethod
    def transpose_axiom(category):
        x, y = map(category.ob, "xy")
        return box(category, "f", x, y),

    @staticmethod
    def rotate_contravariance(category):
        x, y, z = map(category.ob, "xyz")
        return (box(category, "f", x, y), box(category, "g", y, z)),

    @staticmethod
    def hexagon_left(category):
        x, y, z = map(lambda name: Atomic(category.ob(name)), "xyz")
        return x, y, z

    @staticmethod
    def hexagon_right(category):
        x, y, z = map(lambda name: Atomic(category.ob(name)), "xyz")
        return x, y, z

    @staticmethod
    def braid_naturality(category):
        w, x, y, z = map(category.ob, "wxyz")
        return box(category, "f", w, x), box(category, "g", y, z)

    @staticmethod
    def balanced_twist(category):
        x, y = map(lambda name: Atomic(category.ob(name)), "xy")
        return x, y

    @staticmethod
    def swap_inverse(category):
        return tuple(map(category.ob, "xy"))

    @staticmethod
    def copy_counitality(category):
        return category.ob("x"),

    @staticmethod
    def copy_coassociativity(category):
        return category.ob("x"),

    @staticmethod
    def copy_cocommutativity(category):
        return category.ob("x"),

    @staticmethod
    def discard_coherence(category):
        return category.ob("x"),

    @staticmethod
    def copy_monoidal_coherence(category):
        return category.ob("x"),

    @staticmethod
    def feedback_vanishing(category):
        x, y = map(category.ob, "xy")
        return (box(category, "f", x, y), category.ob()),

    @staticmethod
    def feedback_joining(category):
        x, y, z = map(category.ob, "xyz")
        memory = y @ z
        f = category.id(x @ memory.delay()) >> box(
            category, "f", x @ memory.delay(), x @ memory)
        return (f, memory),

    @staticmethod
    def twist_as_trace(category):
        return Atomic(category.ob("x")),

    @staticmethod
    def reidemeister_1_cap(category):
        return category.ob("x"),

    @staticmethod
    def reidemeister_1_cup(category):
        return category.ob("x"),

    @staticmethod
    def frobenius(category):
        return category.ob("x"),

    @staticmethod
    def speciality(category):
        return category.ob("x"),

    @staticmethod
    def spider_fusion(category):
        return category.ob("x"), 1, 2


FREE = {
    abc.Category: cat.Arrow,
    abc.ColouredMonoid: monoidal.Ty,
    abc.MonoidalCategory: monoidal.Diagram,
    abc.TracedCategory: traced.Diagram,
    abc.BiclosedCategory: biclosed.Diagram,
    abc.Pregroup: rigid.Ty,
    abc.RigidCategory: rigid.Diagram,
    abc.PivotalCategory: pivotal.Diagram,
    abc.BraidedCategory: braided.Diagram,
    abc.BalancedCategory: balanced.Diagram,
    abc.SymmetricCategory: symmetric.Diagram,
    abc.MarkovCategory: markov.Diagram,
    abc.ClosedCategory: closed.Diagram,
    abc.FeedbackCategory: feedback.Diagram,
    abc.RibbonCategory: ribbon.Diagram,
    abc.CompactCategory: compact.Diagram,
    abc.HypergraphCategory: frobenius.Diagram,
}


def marks(axiom):
    """ Skip an axiom that does not apply, expect a broken one to fail. """
    if not axiom.parameters and axiom() is NotImplemented:
        return pytest.mark.skip,
    return (pytest.mark.xfail, ) if axiom.broken else ()


def all_axioms():
    """ Every axiom of each structure, as its free category states it. """
    for structure, free_category in FREE.items():
        for axiom in structure.axioms:
            axiom = declared_axioms(free_category).get(
                axiom.name, axiom).bind(free_category)
            yield pytest.param(
                axiom, marks=marks(axiom),
                id=f"{utils.factory_name(free_category)}.{axiom.name}")


@pytest.mark.parametrize("axiom", all_axioms())
def test_axioms_instantiation_on_diagrams(axiom):
    arguments = getattr(Arguments, axiom.name)(axiom.carrier)
    assert_verdict(axiom, axiom(*arguments))


FUNCTORS = [
    cat.Functor, monoidal.Functor, braided.Functor, traced.Functor,
    balanced.Functor, symmetric.Functor, biclosed.Functor, rigid.Functor,
    pivotal.Functor, ribbon.Functor, compact.Functor, markov.Functor,
    closed.Functor, feedback.Functor, frobenius.Functor,
]


def functor_axioms():
    """ Every axiom each level of Functor states or inherits. """
    for functor in FUNCTORS:
        for axiom in functor.axioms:
            yield pytest.param(
                axiom, marks=marks(axiom),
                id=f"{utils.factory_name(functor)}.{axiom.name}")


def functor_arguments(axiom, identity):
    """ Canonical arguments for a law a Functor states or inherits. """
    if axiom.name == "unitality":
        return identity,
    if axiom.name == "associativity":
        return ComposableTriple(identity, identity, identity),
    if axiom.name.startswith("composition"):
        return ComposablePair(identity, identity),
    if not axiom.is_method:
        return ()
    category = axiom.carrier.dom
    x, y = map(category.ob, "xy")
    if axiom.name == "tensor":
        return identity, HorizontalPair(
            box(category, "f", x, y), box(category, "g", y, x))
    if axiom.name == "swap":
        return identity, Atomic(x), Atomic(y)
    return identity, Atomic(x)


@pytest.mark.parametrize("axiom", functor_axioms())
def test_functor_axioms_hold_for_the_identity(axiom):
    """ Each law of a Functor, on the identity functor of its domain. """
    identity = axiom.carrier(Relabelling(), Relabelled(Relabelling()))
    assert_verdict(axiom, axiom(*functor_arguments(axiom, identity)))

@pytest.mark.parametrize("axiom", all_axioms())
def test_axioms_instantiation_on_diagrams(axiom):
    arguments = getattr(Arguments, axiom.name)(axiom.carrier)
    assert_verdict(axiom, axiom(*arguments))


FUNCTORS = [
    cat.Functor, monoidal.Functor, braided.Functor, traced.Functor,
    balanced.Functor, symmetric.Functor, biclosed.Functor, rigid.Functor,
    pivotal.Functor, ribbon.Functor, compact.Functor, markov.Functor,
    closed.Functor, feedback.Functor, frobenius.Functor,
]


@pytest.mark.parametrize("carrier", [
    symmetric.CMap, compact.CMap, closed.CMap, markov.CMap, frobenius.CMap,
    traced.Diagram])
def test_inapplicable_axioms_declare_themselves(carrier):
    """ Every axiom taking no argument answers that it does not apply. """
    declared = [axiom for axiom in carrier.axioms if not axiom.parameters]
    assert declared
    assert all(axiom() is NotImplemented for axiom in declared)


def test_feedback_signature_allows_inferred_boundaries():
    parameters = signature(abc.FeedbackCategory.feedback).parameters
    assert all(parameters[name].default is None
               for name in ("dom", "cod", "mem"))
