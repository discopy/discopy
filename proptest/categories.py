"""
The categories of the property matrix and their parametrisation.

Every file of the suite quantifies over the same list, so it lives here
rather than in any one of them.
"""

import pytest

from discopy import (
    balanced,
    biclosed,
    braided,
    cat,
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
from discopy.utils import factory_name

CATEGORIES = (
    cat.Arrow,
    monoidal.Ty, monoidal.Nat, monoidal.Diagram,
    braided.Diagram,
    traced.Diagram,
    balanced.Diagram,
    symmetric.Diagram,
    biclosed.Ty, biclosed.Diagram,
    rigid.Ty, rigid.Diagram,
    pivotal.Ty, pivotal.Diagram,
    ribbon.Diagram,
    compact.Diagram,
    markov.Diagram,
    closed.Ty, closed.Diagram,
    feedback.Ty, feedback.Diagram,
    frobenius.Ty, frobenius.Diagram,
)


def category_parameters(classify=lambda category: ()):
    """
    One pytest parameter per category, marked by the given classification,
    a function from a category to its marks, e.g. an expected failure.
    """
    for category in CATEGORIES:
        yield pytest.param(
            category, marks=classify(category), id=factory_name(category))
