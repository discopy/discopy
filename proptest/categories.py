"""
The categories of the property matrix and their parametrisation.

Every file of the suite quantifies over the same list, so it lives here
rather than in any one of them.
"""

import pytest

from discopy import cat
from discopy.utils import factory_name

CATEGORIES = (cat.Arrow, )

#: The serialisable carriers that are not categories of their own, so that
#: the roundtrip laws are checked on the terms themselves rather than only
#: on the arrows that contain them: the objects a boundary is made of, and
#: the boxes whose attributes an arrow only serialises indirectly.
CARRIERS = (cat.Ob, cat.Box)


def category_parameters(classify=lambda category: ()):
    """
    One pytest parameter per category, marked by the given classification,
    a function from a category to its marks, e.g. an expected failure.
    """
    for category in CATEGORIES:
        yield pytest.param(
            category, marks=classify(category), id=factory_name(category))
