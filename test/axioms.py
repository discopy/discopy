""" DisCoPy's property-testing module in action. """

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from hypothesis import find
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample
from pytest import raises

from discopy import biclosed, cat, feedback, monoidal, traced
from discopy.axioms import (
    Atomic,
    Axiom,
    AxiomFailure,
    BoundaryConnected,
    C1,
    ComposablePair,
    ComposableTriple,
    Equation,
    FeedbackJoining,
    FeedbackVanishing,
    Grid,
    HorizontalPair,
    LeftCurrying,
    NonEmpty,
    RightCurrying,
    Square,
    Testable,
    TraceDinaturalityLeft,
    TraceDinaturalityRight,
    TraceNaturalityLeft,
    TraceNaturalityRight,
    TraceSuperposing,
    assert_axioms,
    axiom,
    resolve,
    substitute,
)
from discopy.cat import Arrow, Box, Functor, Ob
from discopy.utils import AxiomError, NamedGeneric


@dataclass(frozen=True)
class Endo(Testable, NamedGeneric["factory"]):
    """ An endomorphism of the factory, the subspace a law is weakened to. """

    value: C1

    def __post_init__(self):
        if self.value.dom != self.value.cod:
            raise ValueError("Expected an endomorphism.")

    @classmethod
    def strategy(cls, **params):
        """Generate an arrow with equal domain and codomain."""
        return resolve(cls.factory, **params).filter(
            lambda arrow: arrow.dom == arrow.cod).map(cls)


class Word(str, Testable["Word"]):
    """ A word with tensor given by concatenation, a monoid to grid. """

    __matmul__ = lambda self, other: Word(str(self) + str(other))

    @classmethod
    def strategy(cls, **params):
        """Generate a word over two letters."""
        return st.text("ab", max_size=3).map(cls)


class Row(Grid):
    """ Two horizontally composable cells. """

    n_rows, n_columns = 1, 2


def test_axioms():
    assert_axioms(Arrow)

    class Classified(Arrow):
        """ A category with a broken law and an inapplicable one. """
        unitality = Arrow.unitality.failing("Never holds.")
        dagger_involution = Arrow.dagger_involution.inapplicable("No dagger.")

    assert_axioms(Classified)


def test_strategy():
    x, y = Ob('x'), Ob('y')
    find(Ob.strategy(), lambda ob: ob.name == "a")
    assert find(Arrow.strategy(dom=x, cod=x), lambda _: True) == Arrow.id(x)
    assert find(Arrow.strategy(dom=x, cod=y), lambda _: True).cod == y
    assert find(Arrow.strategy(dom=x), lambda _: True).dom == x
    assert find(Arrow.strategy(cod=y), lambda _: True).cod == y
    assert find(Box.strategy(dom=x), lambda _: True).dom == x


def test_composable_shapes():
    x, y = Ob('x'), Ob('y')
    f, g = Box('f', x, y), Box('g', y, x)
    assert ComposablePair(f, g) == (f, g)
    assert ComposableTriple(f, g, f) == (f, g, f)
    with raises(ValueError):
        ComposablePair(f)
    with raises(AxiomError):
        ComposablePair(f, f)
    pair = find(resolve(ComposablePair[Arrow]), lambda _: True)
    assert isinstance(pair, ComposablePair) and pair[0].cod == pair[1].dom
    assert Row(Word("a"), Word("b")) == ("a", "b")
    with raises(TypeError):
        resolve(int)
    scope = {"C1": Arrow}
    assert substitute(int, scope) is int
    assert substitute(ComposablePair[Arrow], scope) is ComposablePair[Arrow]
    assert substitute(ComposablePair[C1], scope) is ComposablePair[Arrow]


def test_axiom_binding():
    assert repr(Axiom(lambda cls: NotImplemented)) == "Axiom(<lambda>)"
    assert eval(repr(Arrow.unitality)) == cat.Arrow.unitality
    assert hash(Arrow.unitality) == hash(eval(repr(Arrow.unitality)))
    assert Arrow.unitality != Functor.unitality
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented)()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).falsify()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).strategy()
    assert axiom(lambda cls: NotImplemented).bind(Arrow)() is NotImplemented
    box = Box('f', Ob('x'), Ob('y'))
    assert Arrow.unitality(box)
    broken = Arrow.unitality.weaken(f=Endo[C1]).failing("Never holds.")
    assert broken.subspaces == {"f": Endo[C1]}
    loop = Box('g', Ob('x'), Ob('x'))
    with raises(AxiomFailure) as failure:
        broken(Endo(loop))
    assert failure.value.equation


def test_deferred_annotations():
    """ A law compiled without deferred annotations is refused. """
    namespace, source = {}, "def eager(cls, f: int): return NotImplemented"
    exec(compile(source, "<eager>", "exec", dont_inherit=True), namespace)
    with raises(TypeError, match="__future__"):
        axiom(namespace["eager"])


def test_inapplicable():
    law = Arrow.unitality.inapplicable("No identities to cancel.")
    assert law.name == "unitality"
    assert law.__doc__ == "No identities to cancel."
    assert law() is NotImplemented


def test_modulo():
    law = Arrow.unitality.modulo(lambda term: term.dom).bind(Arrow)
    assert law(Box('f', Ob('x'), Ob('y')))


def test_weaken():
    law = Arrow.unitality.weaken(f=Endo[C1]).bind(Arrow)
    assert law.modulo(lambda term: term).subspaces == law.subspaces
    args = find(law.strategy(), lambda _: True)
    assert isinstance(args[0], Endo) and law(*args)


def test_self_annotation():
    @axiom
    def absorbing(cls, f: Self) -> Equation:
        """ The identity on the domain absorbs into any arrow. """
        return Equation(cls.id(f.dom) >> f, f)

    law = absorbing.bind(Arrow)
    args = find(law.strategy(), lambda _: True)
    assert isinstance(args[0], Arrow) and law(*args)


def test_falsify():
    @axiom
    def trivial(cls, f: C1) -> Equation:
        """ Every arrow is an identity, which a box refutes. """
        return Equation(f, cls.id(f.dom))

    counterexample, = trivial.bind(Arrow).falsify()
    assert isinstance(counterexample, Arrow) and counterexample.inside
    assert Arrow.unitality.failing("Never holds.").falsify()
    with raises(NoSuchExample):
        Arrow.associativity.falsify()


def test_axioms_of_category():
    class Broken(Arrow):
        """ A category declaring an inherited law broken. """
        unitality = Arrow.unitality.failing("Never holds.")

    assert Broken.axioms["unitality"].broken
    assert not Arrow.axioms["unitality"].broken

    class Hidden(Arrow):
        """ Assigning a non-axiom over an inherited law drops it. """
        unitality = None

    assert "unitality" not in Hidden.axioms


def test_object_shapes():
    x, y = monoidal.Ty('x'), monoidal.Ty('y')
    assert Atomic(x).value == x and NonEmpty(x @ y).value == x @ y
    with raises(ValueError):
        Atomic(x @ y)
    with raises(ValueError):
        NonEmpty(monoidal.Ty())
    assert len(find(resolve(Atomic[monoidal.Ty]), lambda _: True).value) == 1
    assert find(resolve(NonEmpty[monoidal.Ty]), lambda _: True).value


def test_boundary_connected():
    x = monoidal.Ty('x')
    f, scalar = monoidal.Box('f', x, x), monoidal.Box('s', x[:0], x[:0])
    assert BoundaryConnected(f).value == f
    assert BoundaryConnected(Square(f, f, f, f)).value == (f, f, f, f)
    with raises(ValueError):
        BoundaryConnected(f @ scalar)
    with raises(ValueError):
        BoundaryConnected(HorizontalPair(f, scalar))
    pair = find(resolve(BoundaryConnected[HorizontalPair[monoidal.Diagram]]),
                lambda pair: all(cell.boxes for cell in pair.value))
    assert all(cell.to_map().is_boundary_connected for cell in pair.value)


def test_trace_shapes():
    x, y = traced.Ty('x'), traced.Ty('y')
    f, g = traced.Box('f', x @ y, x @ y), traced.Box('g', y, y)
    assert TraceSuperposing(f, y) == (f, y)
    with raises(AxiomError):
        TraceSuperposing(traced.Box('h', x, y), y)
    assert TraceNaturalityLeft(f, x, g) == (f, x, g)
    assert TraceNaturalityRight(traced.Box('f', y @ x, y @ x), x, g)
    with raises(ValueError):
        TraceNaturalityLeft(f, y, g)
    assert TraceDinaturalityRight(f, g) == (f, g)
    assert TraceDinaturalityLeft(traced.Box('f', y @ x, y @ x), g)
    with raises(ValueError):
        TraceDinaturalityLeft(f, g)
    for shape in (TraceSuperposing, TraceNaturalityLeft,
                  TraceNaturalityRight, TraceDinaturalityLeft,
                  TraceDinaturalityRight):
        assert isinstance(
            find(resolve(shape[traced.Diagram]), lambda _: True), shape)


def test_currying_shapes():
    x, y = biclosed.Ty('x'), biclosed.Ty('y')
    left, right = biclosed.Diagram.ev(x, y), biclosed.Diagram.ev(x, y, False)
    assert LeftCurrying(left, x, y) == (left, x, y)
    assert RightCurrying(right, x, y) == (right, x, y)
    with raises(ValueError):
        LeftCurrying(right, x, y)
    for shape in (LeftCurrying, RightCurrying):
        assert isinstance(
            find(resolve(shape[biclosed.Diagram]), lambda _: True), shape)


def test_feedback_shapes():
    x, y = feedback.Ty('x'), feedback.Ty('y')
    f = feedback.Box('f', x @ (x @ y).delay(), x @ x @ y)
    assert FeedbackVanishing(f, feedback.Ty()) == (f, feedback.Ty())
    with raises(ValueError):
        FeedbackVanishing(f, x)
    assert FeedbackJoining(f, x @ y) == (f, x @ y)
    with raises(ValueError):
        FeedbackJoining(f, y)
    with raises(ValueError):
        FeedbackJoining(f, y @ x)
    with raises(ValueError):
        FeedbackJoining(feedback.Box('g', x @ (x @ y).delay(), x @ y @ x), x @ y)
    for shape in (FeedbackVanishing, FeedbackJoining):
        assert isinstance(
            find(resolve(shape[feedback.Diagram]), lambda _: True), shape)
