"""
Deterministic replay of recorded counterexamples, the memory of the
property suite: see PROPTEST.md for the recording protocol.
"""

from typing import NamedTuple

import pytest

from discopy import biclosed, braided, cat, compact, feedback, pivotal, ribbon
from discopy.python import finset
from discopy.testing import (
    GENERATORS, Atomic, Axiom, Natural, Relabelled, Relabelling)
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
    (cat.Ob(name), cat.Ob("a")) for name in GENERATORS))
"""
The relabelling the search shrunk to: every generator sent to the first.

It names all of them because every functor the strategy builds does, see
:obj:`discopy.testing.GENERATORS`. The images are what shrinking landed on
rather than what the bug needs — composing on the left forgets the functor
whatever it relabels, so the identity relabelling is a counterexample too.
"""

MEMORY = feedback.Ty("a") @ feedback.Ty("b")

COUNTEREXAMPLES = (
    Counterexample(
        axiom=cat.Functor.unitality,
        args=(cat.Functor(ob_map=COLLAPSE, ar_map=Relabelled(COLLAPSE)), ),
        reason="MappingOrCallable.then iterates the keys of the left-hand "
               "map and the identity functor enumerates none, so id >> f "
               "forgets everything f does."),
    Counterexample(
        axiom=braided.Diagram.braid_naturality,
        args=(braided.Box("f", braided.Ty("a"), braided.Ty("a")),
              braided.Box("g", braided.Ty("a"), braided.Ty("a"))),
        reason="A free braid is a box, so naturality only holds up to the "
               "braid relations that free diagrams do not quotient by."),
    Counterexample(
        axiom=biclosed.Diagram.currying_left,
        args=((biclosed.Eval(biclosed.Ty("a") << biclosed.Ty("a")),
               biclosed.Ty("a"), biclosed.Ty("a")), ),
        reason="A free currying is a bubble, equal to its evaluation only "
               "semantically."),
    Counterexample(
        axiom=biclosed.Diagram.currying_right,
        args=((biclosed.Eval(
                   biclosed.Ty("a") >> biclosed.Ty("a"), left=False),
               biclosed.Ty("a"), biclosed.Ty("a")), ),
        reason="A free currying is a bubble, equal to its evaluation only "
               "semantically."),
    Counterexample(
        axiom=pivotal.Diagram.pivotality,
        args=(pivotal.Diagram.id(pivotal.Ty("a")), ),
        reason="The two transposes of a free pivotal diagram are distinct "
               "diagrams, already on the identity wire."),
    Counterexample(
        axiom=ribbon.Diagram.twist_as_trace,
        args=(Atomic(pivotal.Ty("a")), ),
        reason="A free twist is a box, not the trace of a braid."),
    Counterexample(
        axiom=compact.Diagram.rotate_contravariance,
        args=((compact.Box("f", compact.Ty("a"), compact.Ty("a")),
               compact.Box("g", compact.Ty("a"), compact.Ty("a"))), ),
        reason="to_hypergraph drops the rotation of a box, so the equation "
               "holds but cannot be checked up to hypergraph."),
    Counterexample(
        axiom=feedback.Diagram.feedback_joining,
        args=((feedback.Box(
                   "f", MEMORY[:1] @ MEMORY.delay(), MEMORY[:1] @ MEMORY),
               MEMORY), ),
        reason="feedback.Diagram.feedback unrolls its memory in the wrong "
               "order (#606)"),
    Counterexample(
        axiom=finset.Function.hexagon_left,
        args=(Atomic(Natural(1)), Atomic(Natural(1)), Atomic(Natural(1))),
        reason="finset.Function.swap returns the inverse permutation "
               "(#657)"),
    Counterexample(
        axiom=finset.Function.hexagon_right,
        args=(Atomic(Natural(1)), Atomic(Natural(1)), Atomic(Natural(1))),
        reason="finset.Function.swap returns the inverse permutation "
               "(#657)"),
    Counterexample(
        axiom=finset.Function.braid_naturality,
        args=(finset.Function(inside=[0], dom=1, cod=1),
              finset.Function(inside=[], dom=2, cod=0)),
        reason="finset.Function.swap returns the inverse permutation "
               "(#657)"),
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
    assert axiom(*args)
