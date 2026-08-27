"""
Property tests for the consistency of equality and hashing: whiskering by
the monoidal unit preserves a value, its hash and its dictionary lookups.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discopy import cmap, hypergraph, monoidal
from discopy.utils import factory_name

from proptest.test_axioms import CARRIERS

MONOIDS = tuple(
    carrier for carrier in CARRIERS if isinstance(carrier, type) and
    issubclass(carrier, (
        monoidal.Ty, monoidal.Diagram, hypergraph.Hypergraph, cmap.CMap)))


@pytest.mark.parametrize("carrier", MONOIDS, ids=factory_name)
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_eq_hash(carrier, data):
    """ Check that the unit whiskering is invisible to ``==`` and ``hash``. """
    value = data.draw(carrier.strategy())
    whiskered = value @ carrier.id()
    assert value == whiskered and whiskered == value
    assert hash(value) == hash(whiskered)
    assert {value: 0}[whiskered] == 0
