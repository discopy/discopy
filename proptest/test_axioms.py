""" Property tests for DisCoPy's principal categorical data structures. """

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from discopy.axioms import Testable
from discopy.utils import factory_name


def types() -> tuple[type[Testable], ...]:
    """
    The testable types the matrix quantifies over: every transitive
    subclass of :class:`discopy.axioms.Testable` that generates its own
    terms, i.e. every one whose
    :meth:`discopy.axioms.Testable.strategy` is implemented rather than
    left to raise.

    Implementing a strategy is how a type enrols itself, so that the
    matrix follows the package rather than a list kept beside it: a type
    whose terms cannot be generated yet states its laws without being
    checked against them, and is checked as soon as it says how.

    Importing :mod:`discopy.axioms` imports the package that defines
    them, so every subclass is in place by the time this is called.
    :class:`Testable` itself states no law, so it gets no cell.
    """
    def generates(testable):
        try:
            testable.strategy()
        except NotImplementedError:
            return False
        return True

    return tuple(sorted(
        filter(generates, Testable.subclasses()), key=factory_name))


def axiom_parameters():
    """
    Translate every axiom of every testable type to a pytest parameter.

    An axiom taking no argument states its verdict without one, so we ask it
    here: :obj:`NotImplemented` means the structure does not apply and the
    test is skipped rather than generating arguments it could not satisfy.
    A type need not state laws at all: one enrolled for the ad-hoc
    properties only, such as a type of wires, has no ``axioms``.
    """
    for testable in types():
        for axiom in testable.axioms.values():
            if not axiom.parameters and axiom() is NotImplemented:
                marks = pytest.mark.skip(reason=axiom.__doc__.strip())
            elif axiom.broken:
                marks = pytest.mark.xfail(reason=axiom.__doc__.strip())
            else:
                marks = ()
            yield pytest.param(
                axiom, marks=marks,
                id=f"{factory_name(testable)}.{axiom.name}")


@pytest.mark.parametrize("axiom", axiom_parameters())
@given(data=st.data())
def test_axiom(axiom, data):
    """ Check an axiom of a testable type against generated arguments. """
    args = data.draw(axiom.strategy(), label=axiom.name)
    verdict = axiom(*args)
    note(verdict)
    assert verdict
