# -*- coding: utf-8 -*-

"""
Tests for :mod:`discopy.abc`, i.e. the properties every level of the
hierarchy shares. Each module's own axioms are checked on one generated
example by the ``test_axioms`` in its test file, and the search over many
is run by every category in ``proptest/``.
"""

from inspect import signature

import pytest

from discopy import (
    abc, balanced, biclosed, braided, closed, compact, feedback,
    frobenius, markov, monoidal, pivotal, ribbon, rigid, symmetric, traced)


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


def test_strict_equality_is_on_the_nose():
    x, y = map(symmetric.Ty, "xy")
    f, g = symmetric.Box('f', x, x), symmetric.Box('g', y, y)
    left, right = f @ y >> x @ g, x @ g >> f @ y
    assert left != right
    assert not abc.Category.equation_factory(left, right)
    assert symmetric.Diagram.equation_factory(left, right)


@pytest.mark.parametrize("module", [
    monoidal, braided, traced, balanced, symmetric, markov, biclosed,
    closed, rigid, pivotal, ribbon, compact, feedback, frobenius])
def test_every_diagram_level_inherits_its_box_factory(module):
    assert module.Diagram.box_factory is module.Box
