"""
Property tests for the conversions between diagrams, hypergraphs and maps.

Each representation has a canonical decoder ``to_diagram``: the roundtrip
through a diagram must land back on the representation it started from, i.e.
``to_diagram`` is a section of ``to_hypergraph`` and of ``to_map``, and every
conversion preserves ``dom`` and ``cod``.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discopy import (
    balanced,
    biclosed,
    braided,
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

NO_SWAPS = pytest.mark.xfail(reason=(
    "Decoding a trace, cup or cap can cross wires, "
    "which needs swaps the category does not have."))


def levels(*modules, xfail=()):
    """ Translate the levels of the hierarchy to pytest parameters. """
    return tuple(
        pytest.param(
            module, id=module.__name__.removeprefix("discopy."),
            marks=NO_SWAPS if module in xfail else ())
        for module in modules)


HYPERGRAPH_LEVELS = levels(
    monoidal, traced, balanced, symmetric, pivotal, compact, markov,
    closed, feedback, frobenius, xfail=(traced, balanced, pivotal))

CMAP_LEVELS = levels(
    monoidal, traced, balanced, symmetric, biclosed, pivotal, compact,
    markov, closed, frobenius, xfail=(traced, balanced, pivotal))

COMMON_LEVELS = levels(monoidal, traced, balanced, symmetric, pivotal, compact)
"""
The levels whose two encodings agree: ``markov``, ``closed`` and ``frobenius``
are left out because ``to_hypergraph`` encodes their copies and spiders as
spiders while ``to_map`` keeps them as boxes.
"""

ALL_LEVELS = levels(
    monoidal, braided, traced, balanced, symmetric, biclosed, rigid,
    pivotal, ribbon, compact, markov, closed, feedback, frobenius)

SYMMETRIC_LEVELS = levels(
    symmetric, compact, markov, closed, feedback, frobenius)


@pytest.mark.parametrize("module", HYPERGRAPH_LEVELS)
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_hypergraph_section(module, data):
    """ ``to_diagram`` is a section of ``to_hypergraph``. """
    diagram = data.draw(module.Diagram.strategy())
    graph = diagram.to_hypergraph()
    assert (graph.dom, graph.cod) == (diagram.dom, diagram.cod)
    decoded = graph.to_diagram()
    assert (decoded.dom, decoded.cod) == (diagram.dom, diagram.cod)
    assert decoded.to_hypergraph() == graph


@pytest.mark.parametrize("module", CMAP_LEVELS)
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_cmap_section(module, data):
    """ ``to_diagram`` is a section of ``to_map``. """
    diagram = data.draw(module.Diagram.strategy())
    map_ = diagram.to_map()
    assert (map_.dom, map_.cod) == (diagram.dom, diagram.cod)
    decoded = map_.to_diagram()
    assert (decoded.dom, decoded.cod) == (diagram.dom, diagram.cod)
    assert decoded.to_map() == map_


@pytest.mark.parametrize("module", COMMON_LEVELS)
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_cmap_hypergraph_agreement(module, data):
    """
    Encoding through a map or directly gives the same hypergraph — on
    the boundary-connected subspace for pivotal diagrams, whose
    ``to_hypergraph`` rejects a disconnected diagram by design.
    """
    diagram = data.draw(module.Diagram.strategy(
        boundary_connected=module is pivotal))
    assert diagram.to_map().to_hypergraph() == diagram.to_hypergraph()


@pytest.mark.parametrize("module", ALL_LEVELS)
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_encode_decode(module, data):
    """
    ``decode`` undoes ``encode`` up to splitting layers in staircases,
    which decompose plumbing into swaps from symmetric categories on, so
    the comparison is up to the level's own equation.
    """
    diagram = data.draw(module.Diagram.strategy())
    assert module.Diagram.equation_factory(
        type(diagram).decode(*diagram.encode()), diagram.to_staircases())


@pytest.mark.parametrize("module", SYMMETRIC_LEVELS)
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_permutation(module, data):
    """ A permutation is inverted by its dagger and encoded by its swaps. """
    typ = data.draw(module.Ty.strategy())
    perm = module.Diagram.permutation(
        data.draw(st.permutations(range(len(typ)))), typ)
    assert (perm >> perm.dagger()).to_hypergraph()\
        == module.Diagram.id(typ).to_hypergraph()
    for box in perm.boxes:
        assert box.to_swaps().to_hypergraph() == box.to_hypergraph()
