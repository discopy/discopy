"""
Property tests for tree serialisation: every carrier that implements
``to_tree`` decodes back to itself, both through raw trees and through JSON.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discopy import hopf
from discopy.quantum import circuit
from discopy.utils import dumps, from_tree, loads, factory_name

from proptest.test_axioms import CARRIERS


def carrier_parameters():
    """ One parameter per carrier, skipping those without ``to_tree``. """
    for carrier in CARRIERS:
        if not hasattr(carrier, "to_tree"):
            marks = pytest.mark.skip(reason="No tree serialisation.")
        elif carrier is circuit.Circuit:
            marks = pytest.mark.xfail(reason=(
                "Complex gate data does not serialise to JSON."))
        elif carrier is hopf.Intertwiner[
                hopf.Double(hopf.Algebra.cyclic(2))]:
            marks = pytest.mark.skip(reason=(
                "A class subscripted by an algebra instance has no "
                "importable factory name."))
        else:
            marks = ()
        yield pytest.param(carrier, marks=marks, id=factory_name(carrier))


@pytest.mark.parametrize("carrier", carrier_parameters())
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_serialisation(carrier, data):
    """ Check that a value decodes back from its tree and its JSON. """
    value = data.draw(carrier.strategy())
    assert from_tree(value.to_tree()) == value
    assert loads(dumps(value)) == value
