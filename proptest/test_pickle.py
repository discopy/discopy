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

from proptest.test_axioms import carrier_parameters


def classify(carrier):
    """ The closures expected failures. """
    if isinstance(carrier, type)\
            and issubclass(carrier, function.Function):
        return pytest.mark.xfail(reason="A closure does not pickle.")
    return ()


@pytest.mark.parametrize("carrier", carrier_parameters(classify))
@given(data=st.data())
@settings(max_examples=25, deadline=None)
def test_pickle(carrier, data):
    """ Check that a pickled value loads back equal, with the same class. """
    value = data.draw(carrier.strategy())
    loaded = pickle.loads(pickle.dumps(value))
    assert type(loaded) is type(value)
    assert loaded == value
