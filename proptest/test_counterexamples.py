"""
Deterministic replay of recorded counterexamples, the memory of the
property suite: :mod:`discopy.axioms` documents the recording protocol.
"""

from typing import NamedTuple

import pytest

from discopy.axioms import Axiom, AxiomFailure
from discopy.utils import AxiomError, factory_name


class Counterexample(NamedTuple):
    """
    A counterexample once found against a law: the bound axiom itself and
    the very arguments the search shrunk the failure to.
    """
    axiom: Axiom
    args: tuple
    reason: str


COUNTEREXAMPLES: tuple[Counterexample, ...] = ()
"""
The records: none while no law of an enrolled category is declared broken.
"""


def counterexample_parameters():
    """
    One parameter per record, a strict xfail while its axiom is declared
    broken: the day the bug is fixed the record fails as an unexpected pass
    until the ``.failing`` declaration moves.
    """
    for axiom, args, reason in COUNTEREXAMPLES:
        marks = pytest.mark.xfail(
            reason=reason, raises=(AssertionError, AxiomError), strict=True)\
            if axiom.broken else ()
        yield pytest.param(
            axiom, args, marks=marks,
            id=f"{factory_name(axiom.category)}.{axiom.name}")


@pytest.mark.parametrize("axiom, args", counterexample_parameters())
def test_counterexample(axiom, args):
    """
    Check an axiom on a recorded counterexample.

    A broken axiom's failure carries the equation, which the record must
    falsify: its cell xfails while the bug stands and passes — visibly,
    as an expected pass — the day the bug is fixed.
    """
    try:
        assert axiom(*args)
    except AxiomFailure as failure:
        assert failure.equation
