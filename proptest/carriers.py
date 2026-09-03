"""
The carriers of the property matrix and their parametrisation.

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
from discopy.python import finset
from discopy.utils import factory_name

CARRIERS = (
    cat.Arrow, cat.Functor,
    monoidal.Wire, monoidal.Ty, monoidal.PRO,
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
    finset.Function, finset.Permutation,
)


def carrier_parameters(classify=lambda carrier: ()):
    """
    One pytest parameter per carrier, marked by the given classification,
    a function from a carrier to its marks, e.g. an expected failure.
    """
    for carrier in CARRIERS:
        yield pytest.param(
            carrier, marks=classify(carrier), id=factory_name(carrier))
