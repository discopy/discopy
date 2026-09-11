"""
Property tests for the roundtrip laws of :class:`discopy.abc.Serialisable`.

The categories of the matrix inherit these laws with the rest and are
checked in :mod:`proptest.test_axioms`; what is left here are the carriers
that state them without being categories, e.g. the objects of a category.
"""

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from discopy.abc import Serialisable
from discopy.axioms import declared_axioms
from discopy.utils import factory_name

from proptest.categories import CARRIERS

#: The laws to check, i.e. those a serialisable term states about itself.
ROUNDTRIPS = tuple(declared_axioms(Serialisable))


def roundtrip_parameters():
    """ One pytest parameter per roundtrip law of each carrier. """
    for carrier in CARRIERS:
        for name in ROUNDTRIPS:
            axiom = declared_axioms(carrier)[name]
            yield pytest.param(
                axiom, marks=pytest.mark.xfail(
                    reason=axiom.__doc__.strip()) if axiom.broken else (),
                id=f"{factory_name(carrier)}.{name}")


@pytest.mark.parametrize("axiom", roundtrip_parameters())
@given(data=st.data())
def test_roundtrip(axiom, data):
    """ Check that a term reads back from what it was written to. """
    args = data.draw(axiom.strategy(), label=axiom.name)
    verdict = axiom(*args)
    note(verdict)
    assert verdict
