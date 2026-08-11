# -*- coding: utf-8 -*-

"""
Tests for :mod:`discopy.abc`, i.e. that the axioms declared by each abstract
category can be instantiated in the free category that implements it.

Whether the equations hold is checked over every category in ``proptest/``.
"""

import pytest

from discopy import (
    abc, balanced, biclosed, braided, cat, closed, compact, feedback,
    frobenius, markov, monoidal, pivotal, ribbon, rigid, symmetric, traced)
from discopy.testing import Atomic, NonEmpty


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
        x, y = map(category.ob, "xy")
        return (box(category, "f", y @ x, x @ x), x,
                box(category, "g", x, y)),

    @staticmethod
    def trace_dinaturality_right(category):
        x, y = map(category.ob, "xy")
        return (box(category, "f", x @ y, x @ x), x,
                box(category, "g", x, y)),

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
        x, y = map(category.ob, "xy")
        memory = y @ y
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


@pytest.mark.parametrize("Diagram, category", [
    (cat.Arrow, abc.Category),
    (monoidal.Ty, abc.ColouredMonoid),
    (monoidal.Diagram, abc.MonoidalCategory),
    (traced.Diagram, abc.TracedCategory),
    (biclosed.Diagram, abc.BiclosedCategory),
    (rigid.Ty, abc.Pregroup),
    (rigid.Diagram, abc.RigidCategory),
    (pivotal.Diagram, abc.PivotalCategory),
    (braided.Diagram, abc.BraidedCategory),
    (balanced.Diagram, abc.BalancedCategory),
    (symmetric.Diagram, abc.SymmetricCategory),
    (markov.Diagram, abc.MarkovCategory),
    (closed.Diagram, abc.ClosedCategory),
    (feedback.Diagram, abc.FeedbackCategory),
    (ribbon.Diagram, abc.RibbonCategory),
    (compact.Diagram, abc.CompactCategory),
    (frobenius.Diagram, abc.HypergraphCategory),
])
def test_axioms_instantiation_on_diagrams(Diagram, category):
    for axiom in category.axioms:
        arguments = getattr(Arguments, axiom.name)(Diagram)
        assert axiom.bind(Diagram)(*arguments), axiom.name
