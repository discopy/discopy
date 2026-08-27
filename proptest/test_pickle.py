"""
Property tests for serialisation: pickling roundtrips every carrier of the
property matrix, preserving both the value and its class — in particular the
type parameter of a :class:`discopy.abc.NamedGeneric` subscript.
"""

import pickle

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discopy.python import function
from discopy.utils import factory_name

from proptest.test_axioms import CARRIERS


def carrier_parameters():
    """ One parameter per carrier, the closures expected failures. """
    for carrier in CARRIERS:
        marks = pytest.mark.xfail(
            reason="A closure does not pickle.") if isinstance(carrier, type)\
            and issubclass(carrier, function.Function) else ()
        yield pytest.param(carrier, marks=marks, id=factory_name(carrier))


@pytest.mark.parametrize("carrier", carrier_parameters())
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_pickle(carrier, data):
    """ Check that a pickled value loads back equal, with the same class. """
    value = data.draw(carrier.strategy())
    loaded = pickle.loads(pickle.dumps(value))
    assert type(loaded) is type(value)
    assert loaded == value
