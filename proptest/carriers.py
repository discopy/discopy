"""
The carriers of the property matrix and their parametrisation.

Every file of the suite quantifies over the same list, so it lives here
rather than in any one of them.
"""

import pytest

from discopy.axioms import Testable, Theory
from discopy.utils import factory_name


def carriers() -> tuple[type[Theory], ...]:
    """
    The carriers of the matrix: every transitive subclass of
    :class:`discopy.axioms.Theory` that says how to generate its own
    instances, i.e. that defines a ``strategy`` rather than inheriting
    one from the class it refines.

    Declaring a strategy is how a class enrols itself, so that the matrix
    follows the package rather than a list kept beside it: a category
    whose terms cannot be generated yet states its laws without being
    checked against them, and is checked as soon as it says how.

    Importing :mod:`discopy.axioms` imports the package that defines
    them, so every subclass is in place by the time this is called.
    """
    return tuple(sorted(
        (cls for cls in Theory.theories()
         if issubclass(cls, Testable) and "strategy" in cls.__dict__),
        key=factory_name))


CARRIERS = carriers()


def carrier_parameters(classify=lambda carrier: ()):
    """
    One pytest parameter per carrier, marked by the given classification,
    a function from a carrier to its marks, e.g. an expected failure.
    """
    for carrier in CARRIERS:
        yield pytest.param(
            carrier, marks=classify(carrier), id=factory_name(carrier))
