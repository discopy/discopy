"""
Deterministic replay of recorded counterexamples, the memory of the
property suite: see PROPTEST.md for the recording protocol.
"""

from typing import NamedTuple

import pytest

from discopy import cat
from discopy.testing import Axiom, Relabelled, Relabelling, assert_verdict
from discopy.utils import factory_name


class Counterexample(NamedTuple):
    """
    A counterexample once found against a law: the bound axiom itself and
    the very arguments the search shrunk the failure to.
    """
    axiom: Axiom
    args: tuple
    reason: str


COLLAPSE = Relabelling(tuple(
    (cat.Ob(name), cat.Ob("a")) for name in "abcde"))


COUNTEREXAMPLES = (
    Counterexample(
        axiom=cat.Functor.unitality,
        args=(cat.Functor(ob_map=COLLAPSE, ar_map=Relabelled(COLLAPSE)), ),
        reason="MappingOrCallable.then iterates the keys of the left-hand "
               "map and the identity functor enumerates none, so id >> f "
               "forgets everything f does."),
)


def counterexample_parameters():
    """ One parameter per record, xfail while its axiom is declared broken. """
    for axiom, args, reason in COUNTEREXAMPLES:
        marks = pytest.mark.xfail(reason=reason) if axiom.broken else ()
        yield pytest.param(
            axiom, args, marks=marks,
            id=f"{factory_name(axiom.carrier)}.{axiom.name}")


@pytest.mark.parametrize("axiom, args", counterexample_parameters())
def test_counterexample(axiom, args):
    """ Check an axiom on a recorded counterexample. """
    assert_verdict(axiom, axiom(*args))
