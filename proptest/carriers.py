"""
The carriers of the property matrix and their parametrisation.

Every file of the suite quantifies over the same list, so it lives here
rather than in any one of them.
"""

import pytest

from discopy import cat
from discopy.utils import factory_name

CARRIERS = (
    cat.Arrow, cat.Functor,
)


def carrier_parameters(classify=lambda carrier: ()):
    """
    One pytest parameter per carrier, marked by the given classification,
    a function from a carrier to its marks, e.g. an expected failure.
    """
    for carrier in CARRIERS:
        yield pytest.param(
            carrier, marks=classify(carrier), id=factory_name(carrier))
