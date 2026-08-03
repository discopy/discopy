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
from discopy.python import finset
from proptest import strategies

KNOWN_FAILURES = {
    "biclosed.Diagram.currying_left",
    "biclosed.Diagram.currying_right",
    "biclosed.CMap.currying_left",
    "biclosed.CMap.currying_right",
    "braided.Diagram.braid_naturality",
    "closed.Diagram.currying_left",
    "closed.Diagram.currying_right",
    "closed.CMap.currying_left",
    "closed.CMap.currying_right",
    "ribbon.Diagram.braid_naturality",
    "ribbon.Diagram.twist_as_trace",
    "rigid.Diagram.currying_left",
    "rigid.Diagram.currying_right",
    "rigid.Diagram.snake_equations",

    # These ones are due to ``BalancedCategory`` inheriting
    # from ``TracedCategory``.
    "python.finset.Permutation.trace_vanishing",
    "python.finset.Permutation.trace_superposing_left",
    "python.finset.Permutation.trace_superposing_right",
    "python.finset.Permutation.trace_naturality_left",
    "python.finset.Permutation.trace_naturality_right",
    "python.finset.Permutation.trace_dinaturality_left",
    "python.finset.Permutation.trace_dinaturality_right",
}


def axiom_parameter(cls, axiom):
    """Translate an axiom and known failures to a pytest parameter."""
    identifier = ".".join((
        cls.__module__.removeprefix("discopy."),
        cls.__qualname__, axiom.name))
    marks = pytest.mark.xfail(
        reason=f"{identifier} is a known failure")\
        if identifier in KNOWN_FAILURES else ()
    return pytest.param(axiom, id=axiom.name, marks=marks)


def axiom_list(cls):
    """ Bind the axioms implemented by ``category``. """
    for axiom in getattr(cls, "axioms", ()):
        yield axiom_parameter(cls, axiom)


def axiom_tests(cls):
    @pytest.mark.parametrize("axiom", axiom_list(cls))
    @given(data=st.data())
    @settings(max_examples=25, deadline=None)
    def test(self, axiom, data):
        args = data.draw(strategies.arguments(axiom), label=axiom.name)
        assert axiom(*args)
    return test


class Test_cat:
    test_arrow = axiom_tests(cat.Arrow)


class Test_monoidal:
    test_wire = axiom_tests(monoidal.Wire)
    test_ty = axiom_tests(monoidal.Ty)
    # test_pro = axiom_tests(symmetric.PRO)
    test_diagram = axiom_tests(monoidal.Diagram)
    test_hypergraph = axiom_tests(monoidal.Hypergraph)
    test_cmap = axiom_tests(monoidal.CMap)


class Test_braided:
    test_diagram = axiom_tests(braided.Diagram)


class Test_traced:
    test_diagram = axiom_tests(traced.Diagram)
    test_hypergraph = axiom_tests(traced.Hypergraph)
    test_cmap = axiom_tests(traced.CMap)


class Test_balanced:
    test_diagram = axiom_tests(balanced.Diagram)
    test_hypergraph = axiom_tests(balanced.Hypergraph)


class Test_symmetric:
    test_diagram = axiom_tests(symmetric.Diagram)
    test_hypergraph = axiom_tests(symmetric.Hypergraph)
    test_cmap = axiom_tests(symmetric.CMap)


class Test_biclosed:
    test_ty = axiom_tests(biclosed.Ty)
    test_diagram = axiom_tests(biclosed.Diagram)
    test_cmap = axiom_tests(biclosed.CMap)


class Test_rigid:
    test_ty = axiom_tests(rigid.Ty)
    test_diagram = axiom_tests(rigid.Diagram)


class Test_pivotal:
    test_ty = axiom_tests(pivotal.Ty)
    test_diagram = axiom_tests(pivotal.Diagram)
    test_hypergraph = axiom_tests(pivotal.Hypergraph)


class Test_ribbon:
    test_diagram = axiom_tests(ribbon.Diagram)


class Test_compact:
    test_diagram = axiom_tests(compact.Diagram)
    test_hypergraph = axiom_tests(compact.Hypergraph)
    test_cmap = axiom_tests(compact.CMap)


class Test_markov:
    test_diagram = axiom_tests(markov.Diagram)
    test_hypergraph = axiom_tests(markov.Hypergraph)
    test_cmap = axiom_tests(markov.CMap)


class Test_closed:
    test_ty = axiom_tests(closed.Ty)
    test_diagram = axiom_tests(closed.Diagram)
    test_hypergraph = axiom_tests(closed.Hypergraph)
    test_cmap = axiom_tests(closed.CMap)


class Test_feedback:
    test_ty = axiom_tests(feedback.Ty)
    test_diagram = axiom_tests(feedback.Diagram)
    test_hypergraph = axiom_tests(feedback.Hypergraph)


class Test_frobenius:
    test_ty = axiom_tests(frobenius.Ty)
    test_diagram = axiom_tests(frobenius.Diagram)
    test_hypergraph = axiom_tests(frobenius.Hypergraph)
    test_cmap = axiom_tests(frobenius.CMap)


class Test_finset:
    test_function = axiom_tests(finset.Function)
    test_permutation = axiom_tests(finset.Permutation)
