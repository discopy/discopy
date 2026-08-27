""" Property tests for DisCoPy's principal categorical data structures. """

import pytest
from hypothesis import given, settings
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
    monoidal,
    pivotal,
    ribbon,
    rigid,
    symmetric,
    traced,
)
from discopy.matrix import Matrix
from discopy.python import finset
from discopy.testing import assert_verdict
from discopy.utils import factory_name

CARRIERS = (
    cat.Arrow, cat.Functor,
    monoidal.Wire, monoidal.Ty, monoidal.Diagram, monoidal.Hypergraph,
    monoidal.CMap, monoidal.Functor,
    braided.Diagram, braided.Functor,
    traced.Diagram, traced.Hypergraph, traced.CMap, traced.Functor,
    balanced.Diagram, balanced.Hypergraph, balanced.Functor,
    symmetric.Diagram, symmetric.Hypergraph, symmetric.CMap,
    symmetric.Functor,
    biclosed.Ty, biclosed.Diagram, biclosed.CMap, biclosed.Functor,
    rigid.Ty, rigid.Diagram, rigid.Functor,
    pivotal.Ty, pivotal.Diagram, pivotal.Hypergraph, pivotal.Functor,
    ribbon.Diagram, ribbon.Functor,
    compact.Diagram, compact.Hypergraph, compact.CMap, compact.Functor,
    markov.Diagram, markov.Hypergraph, markov.CMap, markov.Functor,
    closed.Ty, closed.Diagram, closed.Hypergraph, closed.CMap,
    closed.Functor,
    feedback.Ty, feedback.Diagram, feedback.Hypergraph, feedback.Functor,
    frobenius.Ty, frobenius.Diagram, frobenius.Hypergraph, frobenius.CMap,
    frobenius.Functor,
    Matrix[int], finset.Function, finset.Permutation)


def axiom_parameters(broken: bool):
    """
    Translate every axiom of every carrier to a pytest parameter, splitting
    the laws that must hold from the ones declared broken.

    An axiom taking no argument states its verdict without one, so we ask it
    here: :obj:`NotImplemented` means the structure does not apply and the
    test is skipped rather than generating arguments it could not satisfy.
    """
    for carrier in CARRIERS:
        for axiom in getattr(carrier, "axioms", ()):
            if axiom.broken != broken:
                continue
            if not axiom.parameters and axiom() is NotImplemented:
                marks = pytest.mark.skip(reason=axiom.__doc__.strip())
            else:
                marks = ()
            yield pytest.param(
                axiom, marks=marks,
                id=f"{factory_name(carrier)}.{axiom.name}")


@pytest.mark.parametrize("axiom", axiom_parameters(broken=False))
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_axiom(axiom, data):
    """ Check an axiom of a carrier against generated arguments. """
    args = data.draw(axiom.strategy(), label=axiom.name)
    assert_verdict(axiom, axiom(*args))


@pytest.mark.parametrize("axiom", axiom_parameters(broken=True))
def test_broken_axiom(axiom):
    """
    A law declared broken must have a findable counterexample: random
    sampling may miss it and read as an unexplained pass, so we search with
    :meth:`Axiom.falsify`, which raises ``NoSuchExample`` on a declaration
    gone stale.
    """
    axiom.falsify()
