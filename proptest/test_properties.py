""" Property tests for DisCoPy's principal categorical data structures. """

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
from discopy.matrix import Matrix
from discopy.testing import assert_verdict
from discopy.python import finset
from proptest import strategies


def axiom_parameter(axiom):
    """
    Translate an axiom to a pytest parameter.

    An axiom taking no argument states its verdict without one, so we ask it
    here: :obj:`NotImplemented` means the structure does not apply and the
    test is skipped rather than generating arguments it could not satisfy.
    """
    if not axiom.parameters and axiom() is NotImplemented:
        return pytest.param(
            axiom, id=f"{axiom.name} (wontfix)",
            marks=pytest.mark.skip(reason=axiom.__doc__.strip()))
    if axiom.broken:
        return pytest.param(
            axiom, id=f"{axiom.name} (bug)",
            marks=pytest.mark.xfail(reason=axiom.__doc__.strip()))
    return pytest.param(axiom, id=axiom.name)


def axiom_list(cls):
    """ Bind the axioms implemented by ``category``. """
    for axiom in getattr(cls, "axioms", ()):
        yield axiom_parameter(axiom)


def axiom_tests(cls):
    @pytest.mark.parametrize("axiom", axiom_list(cls))
    @given(data=st.data())
    @settings(max_examples=25, deadline=None)
    def test(self, axiom, data):
        args = data.draw(strategies.arguments(axiom), label=axiom.name)
        assert_verdict(axiom, axiom(*args))
    return test


class Test_cat:
    test_arrow = axiom_tests(cat.Arrow)
    test_functor = axiom_tests(cat.Functor)


class Test_monoidal:
    test_wire = axiom_tests(monoidal.Wire)
    test_ty = axiom_tests(monoidal.Ty)
    # test_pro = axiom_tests(symmetric.PRO)
    test_diagram = axiom_tests(monoidal.Diagram)
    test_hypergraph = axiom_tests(monoidal.Hypergraph)
    test_cmap = axiom_tests(monoidal.CMap)
    test_functor = axiom_tests(monoidal.Functor)


class Test_braided:
    test_diagram = axiom_tests(braided.Diagram)
    test_functor = axiom_tests(braided.Functor)


class Test_traced:
    test_diagram = axiom_tests(traced.Diagram)
    test_hypergraph = axiom_tests(traced.Hypergraph)
    test_cmap = axiom_tests(traced.CMap)
    test_functor = axiom_tests(traced.Functor)


class Test_balanced:
    test_diagram = axiom_tests(balanced.Diagram)
    test_hypergraph = axiom_tests(balanced.Hypergraph)
    test_functor = axiom_tests(balanced.Functor)


class Test_symmetric:
    test_diagram = axiom_tests(symmetric.Diagram)
    test_hypergraph = axiom_tests(symmetric.Hypergraph)
    test_cmap = axiom_tests(symmetric.CMap)
    test_functor = axiom_tests(symmetric.Functor)


class Test_biclosed:
    test_ty = axiom_tests(biclosed.Ty)
    test_diagram = axiom_tests(biclosed.Diagram)
    test_cmap = axiom_tests(biclosed.CMap)
    test_functor = axiom_tests(biclosed.Functor)


class Test_rigid:
    test_ty = axiom_tests(rigid.Ty)
    test_diagram = axiom_tests(rigid.Diagram)
    test_functor = axiom_tests(rigid.Functor)


class Test_pivotal:
    test_ty = axiom_tests(pivotal.Ty)
    test_diagram = axiom_tests(pivotal.Diagram)
    test_hypergraph = axiom_tests(pivotal.Hypergraph)
    test_functor = axiom_tests(pivotal.Functor)


class Test_ribbon:
    test_diagram = axiom_tests(ribbon.Diagram)
    test_functor = axiom_tests(ribbon.Functor)


class Test_compact:
    test_diagram = axiom_tests(compact.Diagram)
    test_hypergraph = axiom_tests(compact.Hypergraph)
    test_cmap = axiom_tests(compact.CMap)
    test_functor = axiom_tests(compact.Functor)


class Test_markov:
    test_diagram = axiom_tests(markov.Diagram)
    test_hypergraph = axiom_tests(markov.Hypergraph)
    # test_cmap = axiom_tests(markov.CMap)
    test_functor = axiom_tests(markov.Functor)


class Test_closed:
    test_ty = axiom_tests(closed.Ty)
    test_diagram = axiom_tests(closed.Diagram)
    test_hypergraph = axiom_tests(closed.Hypergraph)
    test_cmap = axiom_tests(closed.CMap)
    test_functor = axiom_tests(closed.Functor)


class Test_feedback:
    test_ty = axiom_tests(feedback.Ty)
    test_diagram = axiom_tests(feedback.Diagram)
    test_hypergraph = axiom_tests(feedback.Hypergraph)
    test_functor = axiom_tests(feedback.Functor)


class Test_frobenius:
    test_ty = axiom_tests(frobenius.Ty)
    test_diagram = axiom_tests(frobenius.Diagram)
    test_hypergraph = axiom_tests(frobenius.Hypergraph)
    # test_cmap = axiom_tests(frobenius.CMap)
    test_functor = axiom_tests(frobenius.Functor)


class Test_matrix:
    test_matrix = axiom_tests(Matrix[int])


class Test_finset:
    test_function = axiom_tests(finset.Function)
    test_permutation = axiom_tests(finset.Permutation)
