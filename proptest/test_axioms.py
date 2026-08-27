""" Property tests for DisCoPy's principal categorical data structures. """

import pytest
from hypothesis import HealthCheck, given, settings
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
from discopy import tensor
from discopy.frobenius import Dim
from discopy.matrix import Matrix
from discopy.quantum import circuit
from discopy.testing import assert_verdict
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
    Matrix[int],
    Dim, tensor.Diagram, tensor.Tensor[int],
    circuit.Circuit,
)


def axiom_parameters():
    """
    Translate every axiom of every carrier to a pytest parameter.

    An axiom taking no argument states its verdict without one, so we ask it
    here: :obj:`NotImplemented` means the structure does not apply and the
    test is skipped rather than generating arguments it could not satisfy.
    """
    for carrier in CARRIERS:
        for axiom in getattr(carrier, "axioms", ()):
            if not axiom.parameters and axiom() is NotImplemented:
                marks = pytest.mark.skip(reason=axiom.__doc__.strip())
            elif axiom.broken:
                marks = pytest.mark.xfail(reason=axiom.__doc__.strip())
            else:
                marks = ()
            yield pytest.param(
                axiom, marks=marks,
                id=f"{factory_name(carrier)}.{axiom.name}")


@pytest.mark.parametrize("axiom", axiom_parameters())
@given(data=st.data())
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.filter_too_much])
def test_axiom(axiom, data):
    """ Check an axiom of a carrier against generated arguments. """
    args = data.draw(axiom.strategy(), label=axiom.name)
    assert_verdict(axiom, axiom(*args))
