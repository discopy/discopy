"""
Property tests for the rewriting methods: ``normal_form`` and ``foliation``
are idempotent and preserve the diagram up to hypergraph.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from discopy import hopf, monoidal, pivotal, ribbon, rigid
from discopy.quantum import zx
from discopy.utils import factory_name

from proptest.categories import CATEGORIES

PARTIAL_HYPERGRAPH = pytest.mark.xfail(reason=(
    "to_hypergraph rejects a left-handed cup or cap: Hypergraph.cups and "
    "caps only accept the right-adjoint orientation."))

WRONG_SPIDER_FACTORY = pytest.mark.xfail(reason=(
    "The functor image of a spider is built by tensor's spider factory, "
    "which expects dimensions rather than PRO types."))

REP_DUALS = pytest.mark.xfail(reason=(
    "The hypergraph functor rebuilds a representation-typed cup or cap "
    "whose adjoint is its dimension reversal, not the dual module."))


def diagram_parameters(xfail=()):
    """ One parameter per diagram category, with per-test expected fails. """
    for category in CATEGORIES:
        if not (isinstance(category, type)
                and issubclass(category, monoidal.Diagram)):
            continue
        if category is rigid.Diagram:
            marks = PARTIAL_HYPERGRAPH
        elif category in xfail:
            marks = xfail[category]
        else:
            marks = ()
        yield pytest.param(category, marks=marks, id=factory_name(category))


DIAGRAMS = tuple(diagram_parameters(xfail={
    hopf.Intertwiner[hopf.Double(hopf.Algebra.cyclic(2))]: REP_DUALS}))


@pytest.mark.parametrize("category", DIAGRAMS)
@given(data=st.data())
def test_normal_form(category, data):
    """
    Check that ``normal_form`` is an idempotent representative, on the
    boundary-connected subspace where it is defined.
    """
    diagram = data.draw(category.strategy(boundary_connected=True))
    normal = diagram.normal_form()
    assert (normal.dom, normal.cod) == (diagram.dom, diagram.cod)
    assert normal.normal_form() == normal
    assert normal.to_hypergraph() == diagram.to_hypergraph()


@pytest.mark.parametrize(
    "category", tuple(diagram_parameters(xfail={
        zx.Diagram: WRONG_SPIDER_FACTORY,
        hopf.Intertwiner[hopf.Double(hopf.Algebra.cyclic(2))]: REP_DUALS})))
@given(data=st.data())
def test_foliation(category, data):
    """
    Check that ``foliation`` is an idempotent representative — on the
    boundary-connected subspace for pivotal and ribbon diagrams, whose
    ``to_hypergraph`` rejects a disconnected diagram by design.
    """
    diagram = data.draw(category.strategy(boundary_connected=category in (
        pivotal.Diagram, ribbon.Diagram)))
    foliated = diagram.foliation()
    assert (foliated.dom, foliated.cod) == (diagram.dom, diagram.cod)
    assert foliated.foliation() == foliated
    assert foliated.to_hypergraph() == diagram.to_hypergraph()
