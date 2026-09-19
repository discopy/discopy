"""
The categories of the property matrix and their parametrisation.

Every file of the suite quantifies over the same list, so it lives here
rather than in any one of them.
"""

import pytest

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
    monoidal,
    pivotal,
    ribbon,
    rigid,
    symmetric,
    traced,
)
from discopy.utils import factory_name

CATEGORIES = (
    cat.Arrow, cat.Functor,
    monoidal.Ty, monoidal.Nat,
    monoidal.Diagram, monoidal.Hypergraph,
    monoidal.CMap, monoidal.Functor,
    braided.Diagram, braided.Functor,
    traced.Diagram, traced.Hypergraph, traced.CMap, traced.Functor,
    balanced.Diagram, balanced.Hypergraph, balanced.CMap, balanced.Functor,
    symmetric.Diagram, symmetric.Hypergraph, symmetric.CMap,
    symmetric.Functor,
    biclosed.Ty, biclosed.Diagram, biclosed.CMap, biclosed.Functor,
    rigid.Ty, rigid.Diagram, rigid.Functor,
    pivotal.Ty, pivotal.Diagram, pivotal.Hypergraph, pivotal.CMap,
    pivotal.Functor,
    ribbon.Diagram, ribbon.Functor,
    compact.Diagram, compact.Hypergraph, compact.CMap, compact.Functor,
    markov.Diagram, markov.Hypergraph, markov.CMap, markov.Functor,
    closed.Ty, closed.Diagram, closed.Hypergraph, closed.CMap,
    closed.Functor,
    feedback.Ty, feedback.Diagram, feedback.Hypergraph, feedback.Functor,
    frobenius.Ty, frobenius.Diagram, frobenius.Hypergraph, frobenius.CMap,
    frobenius.Functor,
)


def category_parameters(classify=lambda category: ()):
    """
    One pytest parameter per category, marked by the given classification,
    a function from a category to its marks, e.g. an expected failure.
    """
    for category in CATEGORIES:
        yield pytest.param(
            category, marks=classify(category), id=factory_name(category))
