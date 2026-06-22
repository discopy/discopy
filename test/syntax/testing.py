# -*- coding: utf-8 -*-

""" Property-based tests for the axioms checked in :mod:`discopy.abc`. """

import random as _random

from discopy import (
    monoidal, braided, balanced, symmetric, rigid, pivotal,
    markov, compact, frobenius, feedback)
from discopy.abc import Monoid, Pregroup
from discopy.testing import (
    random_ty, random_diagram, random_parallel_pair,
    random_composable_triple, check_property)

N_TRIALS = 30


def test_category_associativity_and_unitality():
    def holds(seed):
        f, g, h = random_composable_triple(
            symmetric.Box, symmetric.Ty('x'), seed=seed)
        return f.check_associativity(g, h) and g.check_unitality()
    assert not check_property(holds, n_trials=N_TRIALS)


def test_monoid_unitality_and_associativity():
    def holds(seed):
        rng = _random.Random(seed)
        x, y, z = (
            random_ty(monoidal.Ty, min_length=1, max_length=2, rng=rng)
            for _ in range(3))
        return Monoid.check_monoid_unitality(x, monoidal.Ty())\
            and Monoid.check_monoid_associativity(x, y, z)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_tensor_unitality():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(symmetric.Ty, min_length=0, max_length=2, rng=rng)
        y = random_ty(symmetric.Ty, min_length=0, max_length=2, rng=rng)
        return symmetric.Diagram.check_tensor_unitality(x, y)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_bifunctoriality():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(symmetric.Ty, min_length=1, max_length=2, rng=rng)
        y = random_ty(symmetric.Ty, min_length=1, max_length=2, rng=rng)
        f = random_diagram(symmetric.Box, x, n_boxes=3, rng=rng)
        other = random_diagram(symmetric.Box, y, n_boxes=3, rng=rng)
        g = random_diagram(symmetric.Box, f.cod, n_boxes=3, rng=rng)
        h = random_diagram(symmetric.Box, other.cod, n_boxes=3, rng=rng)
        with symmetric.Diagram.hypergraph_equality:
            return f.check_bifunctoriality(other, g, h)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_pregroup_adjunction():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(rigid.Ty, min_length=1, max_length=3, rng=rng)
        return Pregroup.check_adjunction(x)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_rigid_snake_equations():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(rigid.Ty, min_length=1, max_length=2, rng=rng)
        eq = lambda f, g: f.normal_form() == g.normal_form()
        return rigid.Diagram.check_snake_equations(x, eq=eq)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_pivotal_self_dual():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(pivotal.Ty, min_length=1, max_length=3, rng=rng)
        return pivotal.Diagram.check_self_dual(x)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_braided_hexagon():
    # Multi-object braids are only equal to their hexagon decomposition up
    # to interchange, so we stick to atomic types here.
    def holds(seed):
        rng = _random.Random(seed)
        x, y, z = (
            random_ty(braided.Ty, min_length=1, max_length=1, rng=rng)
            for _ in range(3))
        return braided.Diagram.check_hexagon(x, y, z)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_braid_naturality():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(symmetric.Ty, min_length=1, max_length=2, rng=rng)
        y = random_ty(symmetric.Ty, min_length=1, max_length=2, rng=rng)
        f = random_diagram(symmetric.Box, x, n_boxes=3, rng=rng)
        other = random_diagram(symmetric.Box, y, n_boxes=3, rng=rng)
        with symmetric.Diagram.hypergraph_equality:
            return f.check_braid_naturality(other)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_balanced_twist():
    # As for the hexagon, we stick to atomic types.
    def holds(seed):
        rng = _random.Random(seed)
        x, y = (
            random_ty(balanced.Ty, min_length=1, max_length=1, rng=rng)
            for _ in range(2))
        return balanced.Diagram.check_balanced_twist(x, y)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_swap_inverse_and_yang_baxter():
    def holds(seed):
        rng = _random.Random(seed)
        x, y, z = (
            random_ty(symmetric.Ty, min_length=1, max_length=2, rng=rng)
            for _ in range(3))
        with symmetric.Diagram.hypergraph_equality:
            return symmetric.Diagram.check_swap_inverse(x, y)\
                and symmetric.Diagram.check_yang_baxter(x, y, z)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_trace_vanishing_and_superposing():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(symmetric.Ty, min_length=1, max_length=2, rng=rng)
        obj = random_ty(symmetric.Ty, min_length=1, max_length=2, rng=rng)
        f = random_diagram(symmetric.Box, x, n_boxes=3, rng=rng)
        endo = symmetric.Box('f', x, x)
        with symmetric.Diagram.hypergraph_equality:
            return f.check_trace_vanishing()\
                and endo.check_trace_superposing(obj, left=False)\
                and endo.check_trace_superposing(obj, left=True)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_trace_naturality_and_dinaturality():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(symmetric.Ty, min_length=1, max_length=1, rng=rng)
        f = symmetric.Box('f', x @ x, x @ x)
        g = symmetric.Box('g', x, x)
        with symmetric.Diagram.hypergraph_equality:
            return f.check_trace_naturality(x, g, left=False)\
                and f.check_trace_naturality(x, g, left=True)\
                and f.check_trace_dinaturality(x, g, left=False)\
                and f.check_trace_dinaturality(x, g, left=True)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_markov_copy_axioms():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(markov.Ty, min_length=1, max_length=2, rng=rng)
        with markov.Diagram.hypergraph_equality:
            return markov.Diagram.check_copy_counitality(x)\
                and markov.Diagram.check_copy_coassociativity(x)\
                and markov.Diagram.check_copy_cocommutativity(x)\
                and markov.Diagram.check_copy_coherence(x)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_feedback_vanishing_and_joining():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(feedback.Ty, min_length=1, max_length=1, rng=rng)
        m = random_ty(feedback.Ty, min_length=1, max_length=1, rng=rng)
        mem, unit = m @ m, feedback.Ty()
        f = feedback.Box('f', x @ mem.delay(), x @ mem)
        g = feedback.Box('g', x @ unit.delay(), x @ unit)
        return f.check_feedback_joining(mem)\
            and g.check_feedback_vanishing(unit)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_compact_reidemeister_1_and_caps_coherence():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(compact.Ty, min_length=1, max_length=2, rng=rng)
        y = random_ty(compact.Ty, min_length=1, max_length=2, rng=rng)
        with compact.Diagram.hypergraph_equality:
            return compact.Diagram.check_reidemeister_1(x)\
                and compact.Diagram.check_caps_coherence(x, y)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_frobenius_and_speciality():
    def holds(seed):
        rng = _random.Random(seed)
        x = random_ty(frobenius.Ty, min_length=1, max_length=2, rng=rng)
        with frobenius.Diagram.hypergraph_equality:
            return frobenius.Diagram.check_frobenius(x)\
                and frobenius.Diagram.check_speciality(x)
    assert not check_property(holds, n_trials=N_TRIALS)


def test_random_parallel_pair_shares_domain():
    f, g = random_parallel_pair(symmetric.Box, symmetric.Ty('x'), seed=420)
    assert f.dom == g.dom == symmetric.Ty('x')
