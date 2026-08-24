"""P9: every operation builds a well-typed result or raises AxiomError.
Miniature of the property over curated examples; red while its bullets (B19, B25, B27, B28, B31, B34) are live — issue #606.
"""
from discopy import feedback, monoidal, rigid, stream, symmetric
from discopy.grammar.pregroup import Cup, Ty, Word
from discopy.utils import AxiomError


def _pregroup_normal_form():
    s, n = Ty('s'), Ty('n')
    words = Word('Alice', n) @ Word('loves', n.r @ s @ n.l) @ Word('Bob', n)
    sentence = words >> Cup(n, n.r) @ s @ Cup(n.l, n)
    return sentence.foliation().normal_form()


def _heterogeneous_feedback():
    x, m, n = feedback.Ty('x'), feedback.Ty('m'), feedback.Ty('n')
    f = feedback.Box('f', x @ (m @ n).delay(), x @ m @ n)
    return f.feedback(mem=m @ n)


def test_p9():
    g = symmetric.Box('g', symmetric.Ty('x'), symmetric.Ty('x'))
    cases = [  # (label, thunk, "build" or "axiom")
        ("Id(Ty('x')).width",
         lambda: monoidal.Id(monoidal.Ty('x')).width, "build"),
        ("symmetric trace(n=5) on one wire", lambda: g.trace(n=5), "axiom"),
        ("symmetric trace(n=-1)", lambda: g.trace(n=-1), "axiom"),
        ("rigid cups(Ty(), Ty('n'))",
         lambda: rigid.Diagram.cups(rigid.Ty(), rigid.Ty('n')), "axiom"),
        ("feedback with heterogeneous mem", _heterogeneous_feedback, "build"),
        ("stream permutation swap on two types",
         lambda: stream.Stream.permutation(
             (1, 0), stream.Ty('x') @ stream.Ty('y')), "build"),
        ("pregroup foliated normal_form", _pregroup_normal_form, "build"),
    ]
    failures = []
    for label, thunk, expect in cases:
        try:
            thunk()
            if expect == "axiom":
                failures.append(
                    f"{label}: built instead of raising AxiomError")
        except AxiomError:
            if expect == "build":
                failures.append(
                    f"{label}: raised AxiomError instead of building")
        except Exception as error:
            failures.append(f"{label}: raised {type(error).__name__}")
    assert not failures, failures
