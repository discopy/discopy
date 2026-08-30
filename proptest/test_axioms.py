""" Property tests for DisCoPy's principal categorical data structures. """

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from discopy.utils import factory_name

from proptest.carriers import CARRIERS


def axiom_parameters():
    """
    Translate every axiom of every carrier to a pytest parameter.

    An axiom taking no argument states its verdict without one, so we ask it
    here: :obj:`NotImplemented` means the structure does not apply and the
    test is skipped rather than generating arguments it could not satisfy.
    """
    for carrier in CARRIERS:
        for axiom in getattr(carrier, "axioms", {}).values():
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
    assert axiom(*args)
