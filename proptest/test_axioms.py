""" Property tests for DisCoPy's principal categorical data structures. """

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from discopy.axioms import Theory
from discopy.utils import factory_name


def carriers() -> tuple[type[Theory], ...]:
    """
    The carriers of the matrix: every transitive subclass of
    :class:`discopy.axioms.Theory` that generates its own terms, i.e.
    every one that has not declared
    :data:`discopy.axioms.no_strategy` as its axioms.

    Opting out is how a class stays out of the matrix, so that the
    matrix follows the package rather than a list kept beside it: a
    category whose terms cannot be generated yet states its laws without
    being checked against them, and is checked as soon as it says how.

    Importing :mod:`discopy.axioms` imports the package that defines
    them, so every subclass is in place by the time this is called.
    :class:`Theory` itself states no law, so it carries no cell.
    """
    def generates(carrier):
        try:
            carrier.axioms
        except NotImplementedError:
            return False
        return True

    return tuple(sorted(
        filter(generates, Theory.subclasses()), key=factory_name))


def axiom_parameters():
    """
    Translate every axiom of every carrier to a pytest parameter.

    An axiom taking no argument states its verdict without one, so we ask it
    here: :obj:`NotImplemented` means the structure does not apply and the
    test is skipped rather than generating arguments it could not satisfy.
    A carrier need not state laws at all: one enrolled for the ad-hoc
    properties only, such as a type of wires, has no ``axioms``.
    """
    for carrier in carriers():
        for axiom in carrier.axioms.values():
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
def test_axiom(axiom, data):
    """ Check an axiom of a carrier against generated arguments. """
    args = data.draw(axiom.strategy(), label=axiom.name)
    verdict = axiom(*args)
    note(verdict)
    assert verdict
