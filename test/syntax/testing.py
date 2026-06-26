# -*- coding: utf-8 -*-

""" Property-based tests for the axioms checked in :mod:`discopy.abc`. """

from discopy import (
    monoidal, braided, balanced, symmetric, rigid, pivotal,
    markov, compact, frobenius, feedback)
from pytest import raises

from discopy.abc import Monoid, Pregroup
from discopy.testing import (
    random_ty, random_diagram, random_parallel_pair,
    random_composable_pair, random_composable_triple,
    check_property, Falsified)

N_TRIALS = 30


def atomic(ty):
    """ A generator of atomic types, e.g. for hexagon and balanced twists. """
    return lambda rng: random_ty(ty, min_length=1, max_length=1, rng=rng)


def nonempty(ty):
    """ A generator of non-empty types. """
    return lambda rng: random_ty(ty, min_length=1, max_length=2, rng=rng)


def test_category_associativity_and_unitality():
    assert check_property(
        lambda rng: random_composable_triple(symmetric.Box, rng=rng),
        lambda fgh: fgh[0].check_associativity(fgh[1], fgh[2])
        and fgh[1].check_unitality(), n_trials=N_TRIALS)


def test_category_typing():
    assert check_property(
        lambda rng: (
            random_ty(symmetric.Ty, rng=rng),
            random_composable_pair(symmetric.Box, rng=rng)),
        lambda args: symmetric.Diagram.check_identity_typing(args[0])
        and args[1][0].check_composition_typing(args[1][1]),
        n_trials=N_TRIALS)


def test_monoid_unitality_and_associativity():
    assert check_property(
        lambda rng: tuple(nonempty(monoidal.Ty)(rng) for _ in range(3)),
        lambda xyz: Monoid.check_monoid_unitality(xyz[0], monoidal.Ty())
        and Monoid.check_monoid_associativity(*xyz), n_trials=N_TRIALS)


def test_tensor_unitality_and_typing():
    assert check_property(
        lambda rng: (
            random_ty(symmetric.Ty, max_length=2, rng=rng),
            random_ty(symmetric.Ty, max_length=2, rng=rng),
            random_parallel_pair(symmetric.Box, rng=rng)),
        lambda args: symmetric.Diagram.check_tensor_unitality(args[0], args[1])
        and args[2][0].check_tensor_typing(args[2][1]), n_trials=N_TRIALS)


def test_bifunctoriality():
    def predicate(pairs):
        (f, g), (other, h) = pairs
        with symmetric.Diagram.hypergraph_equality:
            return f.check_bifunctoriality(other, g, h)
    assert check_property(
        lambda rng: (
            random_composable_pair(symmetric.Box, rng=rng),
            random_composable_pair(symmetric.Box, rng=rng)),
        predicate, n_trials=N_TRIALS)


def test_pregroup_adjunction():
    assert check_property(
        lambda rng: random_ty(rigid.Ty, min_length=1, max_length=3, rng=rng),
        Pregroup.check_adjunction, n_trials=N_TRIALS)


def test_rigid_snake_equations():
    eq = lambda f, g: f.normal_form() == g.normal_form()
    assert check_property(
        nonempty(rigid.Ty),
        lambda x: rigid.Diagram.check_snake_equations(x, eq=eq),
        n_trials=N_TRIALS)


def test_rigid_caps_coherence():
    assert check_property(
        lambda rng: (nonempty(rigid.Ty)(rng), nonempty(rigid.Ty)(rng)),
        lambda xy: rigid.Diagram.check_caps_coherence(*xy), n_trials=N_TRIALS)


def test_pivotal_self_dual():
    assert check_property(
        lambda rng: random_ty(pivotal.Ty, min_length=1, max_length=3, rng=rng),
        pivotal.Diagram.check_self_dual, n_trials=N_TRIALS)


def test_braided_hexagon():
    # Multi-object braids are only equal to their hexagon decomposition up
    # to interchange, so we stick to atomic types here.
    assert check_property(
        lambda rng: tuple(atomic(braided.Ty)(rng) for _ in range(3)),
        lambda xyz: braided.Diagram.check_hexagon(*xyz), n_trials=N_TRIALS)


def test_braid_naturality():
    def predicate(fg):
        with symmetric.Diagram.hypergraph_equality:
            return fg[0].check_braid_naturality(fg[1])
    assert check_property(
        lambda rng: (
            random_diagram(symmetric.Box, rng=rng),
            random_diagram(symmetric.Box, rng=rng)),
        predicate, n_trials=N_TRIALS)


def test_balanced_twist():
    # As for the hexagon, we stick to atomic types.
    assert check_property(
        lambda rng: (atomic(balanced.Ty)(rng), atomic(balanced.Ty)(rng)),
        lambda xy: balanced.Diagram.check_balanced_twist(*xy),
        n_trials=N_TRIALS)


def test_swap_inverse():
    def predicate(xy):
        with symmetric.Diagram.hypergraph_equality:
            return symmetric.Diagram.check_swap_inverse(*xy)
    assert check_property(
        lambda rng: (nonempty(symmetric.Ty)(rng), nonempty(symmetric.Ty)(rng)),
        predicate, n_trials=N_TRIALS)


def test_trace_vanishing_and_superposing():
    def generator(rng):
        x = nonempty(symmetric.Ty)(rng)
        obj = nonempty(symmetric.Ty)(rng)
        f = random_diagram(symmetric.Box, x, n_boxes=3, rng=rng)
        return x, obj, f

    def predicate(args):
        x, obj, f = args
        endo = symmetric.Box('f', x, x)
        with symmetric.Diagram.hypergraph_equality:
            return f.check_trace_vanishing()\
                and endo.check_trace_superposing(obj, left=False)\
                and endo.check_trace_superposing(obj, left=True)
    assert check_property(generator, predicate, n_trials=N_TRIALS)


def test_trace_naturality_and_dinaturality():
    def predicate(x):
        f = symmetric.Box('f', x @ x, x @ x)
        g = symmetric.Box('g', x, x)
        with symmetric.Diagram.hypergraph_equality:
            return f.check_trace_naturality(x, g, left=False)\
                and f.check_trace_naturality(x, g, left=True)\
                and f.check_trace_dinaturality(x, g, left=False)\
                and f.check_trace_dinaturality(x, g, left=True)
    assert check_property(atomic(symmetric.Ty), predicate, n_trials=N_TRIALS)


def test_markov_copy_axioms():
    def predicate(x):
        with markov.Diagram.hypergraph_equality:
            return markov.Diagram.check_copy_counitality(x)\
                and markov.Diagram.check_copy_coassociativity(x)\
                and markov.Diagram.check_copy_cocommutativity(x)\
                and markov.Diagram.check_copy_coherence(x)
    assert check_property(nonempty(markov.Ty), predicate, n_trials=N_TRIALS)


def test_feedback_vanishing_and_joining():
    def generator(rng):
        return atomic(feedback.Ty)(rng), atomic(feedback.Ty)(rng)

    def predicate(xm):
        x, m = xm
        mem, unit = m @ m, feedback.Ty()
        f = feedback.Box('f', x @ mem.delay(), x @ mem)
        g = feedback.Box('g', x @ unit.delay(), x @ unit)
        return f.check_feedback_joining(mem)\
            and g.check_feedback_vanishing(unit)
    assert check_property(generator, predicate, n_trials=N_TRIALS)


def test_compact_reidemeister_1():
    def predicate(x):
        with compact.Diagram.hypergraph_equality:
            return compact.Diagram.check_reidemeister_1(x)
    assert check_property(nonempty(compact.Ty), predicate, n_trials=N_TRIALS)


def test_frobenius_and_speciality():
    def predicate(x):
        with frobenius.Diagram.hypergraph_equality:
            return frobenius.Diagram.check_frobenius(x)\
                and frobenius.Diagram.check_speciality(x)
    assert check_property(nonempty(frobenius.Ty), predicate, n_trials=N_TRIALS)


def test_pivotal_transpose():
    # check_transpose builds the two transposes the same way as the library.
    x, y = pivotal.Ty('x'), pivotal.Ty('y')
    f = pivotal.Box('f', x, y)
    assert f.check_transpose() == (
        f.transpose(left=True) == f.transpose(left=False))

    # The pivotal axiom is semantic: it fails on the nose but is witnessed by
    # the pivotal hypergraph, which forgets planarity.
    def predicate(g):
        return g.transpose(left=True).to_hypergraph()\
            == g.transpose(left=False).to_hypergraph()
    assert check_property(
        lambda rng: pivotal.Box(
            'g', nonempty(pivotal.Ty)(rng), nonempty(pivotal.Ty)(rng)),
        predicate, n_trials=N_TRIALS)


def test_random_parallel_pair_is_parallel():
    f, g = random_parallel_pair(symmetric.Box, symmetric.Ty('x'), seed=420)
    assert f.is_parallel(g)


def test_interchanger_shrinks_to_minimal():
    # The interchange law holds only up to interchanger normal form, not on
    # the nose. Shrinking should find the smallest counterexample: a pair of
    # boxes on a single atomic wire 'a'.
    def two_boxes(rng):
        atom = lambda: random_ty(
            monoidal.Ty, min_length=1, max_length=1, rng=rng)
        f = monoidal.Box('f', atom(), atom())
        g = monoidal.Box('g', atom(), atom())
        return f, g

    def interchange_on_the_nose(boxes):
        f, g = boxes
        return f @ g.dom >> f.cod @ g == f.dom @ g >> f @ g.cod

    with raises(Falsified) as error:
        check_property(two_boxes, interchange_on_the_nose)
    f, g = error.value.example
    assert f.dom == f.cod == g.dom == g.cod == monoidal.Ty('a')
    lhs, rhs = f @ g.dom >> f.cod @ g, f.dom @ g >> f @ g.cod
    assert lhs != rhs and lhs.normal_form() == rhs.normal_form()
