"""
Property tests for tree serialisation: every carrier that implements
``to_tree`` decodes back to itself, both through raw trees and through JSON.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discopy import hopf
from discopy.quantum import circuit
from discopy.utils import dumps, from_tree, loads

from proptest.test_axioms import carrier_parameters


def classify(carrier):
    """ Skip the carriers without ``to_tree``, xfail the known violations. """
    if not hasattr(carrier, "to_tree"):
        return pytest.mark.skip(reason="No tree serialisation.")
    if carrier is circuit.Circuit:
        return pytest.mark.xfail(reason=(
            "Complex gate data does not serialise to JSON."))
    if carrier is hopf.Intertwiner[hopf.Double(hopf.Algebra.cyclic(2))]:
        return pytest.mark.skip(reason=(
            "A class subscripted by an algebra instance has no "
            "importable factory name."))
    return ()


@pytest.mark.parametrize("carrier", carrier_parameters(classify))
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_serialisation(carrier, data):
    """ Check that a value decodes back from its tree and its JSON. """
    value = data.draw(carrier.strategy())
    assert from_tree(value.to_tree()) == value
    assert loads(dumps(value)) == value
