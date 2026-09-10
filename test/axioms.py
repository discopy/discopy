""" DisCoPy's property-axioms module in action. """

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from hypothesis import find
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample
from pytest import raises

from discopy import cat
from discopy.axioms import (
    C1,
    Axiom,
    AxiomFailure,
    ComposablePair,
    ComposableTriple,
    Equation,
    Grid,
    Strategy,
    assert_axioms,
    axiom,
    resolve,
    substitute,
)
from discopy.cat import Arrow, Box, Functor, Ob
from discopy.utils import AxiomError, NamedGeneric


from discopy import biclosed, cat, feedback, monoidal, rigid, axioms, traced
from discopy.axioms import (
    C0, C1, Atomic, Axiom, AxiomFailure, Square, BoundaryConnected,
    ComposablePair, ComposableTriple, FeedbackJoining, FeedbackVanishing,
    HomogeneousMemory, HorizontalPair, LeftCurrying, NonEmpty,
    Relabelling, RightCurrying, Subsingleton, TraceDinaturalityLeft,
    TraceDinaturalityRight, TraceNaturalityLeft, TraceNaturalityRight,
    TraceSuperposing, axiom, resolve)
from discopy.utils import AxiomError


@dataclass(frozen=True)
class Endo(Strategy, NamedGeneric["factory"]):
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


class Word(str, Strategy["Word"]):
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


def test_Atomic():
    x, y = map(monoidal.Ty, "xy")
    assert Atomic(x).value == x
    with raises(ValueError):
        Atomic(x @ y)
    find(Atomic[monoidal.Ty].strategy(),
         lambda value: len(value.value) == 1)


def test_NonEmpty():
    x = monoidal.Ty('x')
    assert NonEmpty(x).value == x
    with raises(ValueError):
        NonEmpty(monoidal.Ty())
    find(NonEmpty[monoidal.Ty].strategy(),
         lambda value: len(value.value) > 1)
    nested = find(resolve(NonEmpty[ComposablePair[cat.Arrow]]), lambda _: True)
    assert isinstance(nested.value, ComposablePair)


def test_ComposablePair():
    x, y = map(cat.Ob, "xy")
    f, g = cat.Box('f', x, y), cat.Box('g', y, x)
    assert ComposablePair(f, g) == (f, g)
    with raises(ValueError):
        ComposablePair(f)
    with raises(AxiomError):
        ComposablePair(f, f)
    find(ComposablePair[cat.Arrow].strategy(),
         lambda value: all(term.inside for term in value))


def test_ComposableTriple():
    x, y = map(cat.Ob, "xy")
    f, g = cat.Box('f', x, y), cat.Box('g', y, x)
    assert ComposableTriple(f, g, f) == (f, g, f)
    with raises(AxiomError):
        ComposableTriple(f, f, f)
    find(ComposableTriple[cat.Arrow].strategy(),
         lambda value: all(term.inside for term in value))


def test_HorizontalPair():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert HorizontalPair(f, g) == (f, g)
    with raises(ValueError):
        HorizontalPair(f)
    find(HorizontalPair[monoidal.Diagram].strategy(),
         lambda value: all(term.boxes for term in value))


def test_Bifunctor():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert Square(f, f, g, g) == (f, f, g, g)
    with raises(AxiomError):
        Square(f, f, f, f)
    find(Square[monoidal.Diagram].strategy(),
         lambda value: all(
             value[column].boxes or value[column + 2].boxes
             for column in range(2)))


def test_TraceSuperposing():
    x, y, z = map(traced.Ty, "xyz")
    assert TraceSuperposing(traced.Id(x), y) == (traced.Id(x), y)
    with raises(AxiomError):
        TraceSuperposing(traced.Box('f', x, y), z)
    find(TraceSuperposing[traced.Diagram].strategy(),
         lambda value: len(value[1]) > 1)


def test_TraceNaturalityLeft():
    x, y = map(traced.Ty, "xy")
    f, g = traced.Box('f', x @ y, x @ x), traced.Box('g', x, y)
    assert TraceNaturalityLeft(f, x, g) == (f, x, g)
    with raises(ValueError):
        TraceNaturalityLeft(traced.Id(x @ y), x, traced.Id(x))
    find(TraceNaturalityLeft[traced.Diagram].strategy(),
         lambda value: value[2].dom != value[2].cod)


def test_TraceNaturalityRight():
    x, y = map(traced.Ty, "xy")
    f, g = traced.Box('f', y @ x, x @ x), traced.Box('g', x, y)
    assert TraceNaturalityRight(f, x, g) == (f, x, g)
    with raises(ValueError):
        TraceNaturalityRight(traced.Id(x @ y), x, traced.Id(y))
    find(TraceNaturalityRight[traced.Diagram].strategy(),
         lambda value: value[2].dom != value[2].cod)


def test_TraceDinaturalityLeft():
    x, y, z = map(traced.Ty, "xyz")
    f, g = traced.Box('f', x @ z, y @ z), traced.Box('g', y, x)
    assert TraceDinaturalityLeft(f, g) == (f, g)
    with raises(ValueError):
        TraceDinaturalityLeft(g, f)
    find(TraceDinaturalityLeft[traced.Diagram].strategy(),
         lambda value: value[1].dom != value[1].cod)


def test_TraceDinaturalityRight():
    x, y, z = map(traced.Ty, "xyz")
    f, g = traced.Box('f', z @ x, z @ y), traced.Box('g', y, x)
    assert TraceDinaturalityRight(f, g) == (f, g)
    with raises(ValueError):
        TraceDinaturalityRight(g, f)
    shape = find(TraceDinaturalityRight[traced.Diagram].strategy(),
                 lambda value: value[1].dom != value[1].cod)
    sliding = shape[1]
    assert shape[0].dom[-len(sliding.cod):] == sliding.cod
    assert shape[0].cod[-len(sliding.dom):] == sliding.dom


def test_LeftCurrying():
    x, y = map(biclosed.Ty, "xy")
    evaluation = biclosed.Diagram.ev(x, y, left=True)
    assert LeftCurrying(evaluation, x, y) == (evaluation, x, y)
    with raises(ValueError):
        LeftCurrying(evaluation, y, x)
    find(LeftCurrying[biclosed.Diagram].strategy(),
         lambda value: value[1] != value[2])


def test_RightCurrying():
    x, y = map(biclosed.Ty, "xy")
    evaluation = biclosed.Diagram.ev(x, y, left=False)
    assert RightCurrying(evaluation, x, y) == (evaluation, x, y)
    with raises(ValueError):
        RightCurrying(evaluation, y, x)
    find(RightCurrying[biclosed.Diagram].strategy(),
         lambda value: value[1] != value[2])


def test_FeedbackVanishing():
    x = feedback.Ty('x')
    f, unit = feedback.Box('f', x, x), feedback.Ty()
    assert FeedbackVanishing(f, unit) == (f, unit)
    with raises(ValueError):
        FeedbackVanishing(f, x)
    find(FeedbackVanishing[feedback.Diagram].strategy(),
         lambda value: value[0].boxes)


def test_FeedbackJoining():
    x, y, z = map(feedback.Ty, "xyz")
    memory = y @ z
    f = feedback.Box('f', x @ memory.delay(), x @ memory)
    assert FeedbackJoining(f, memory) == (f, memory)
    with raises(ValueError):
        FeedbackJoining(f, feedback.Ty())
    with raises(ValueError):
        FeedbackJoining(feedback.Box('g', x @ memory, x @ memory), memory)
    with raises(ValueError):
        FeedbackJoining(
            feedback.Box('g', x @ memory.delay(), x @ memory.delay()), memory)
    shape = find(FeedbackJoining[feedback.Diagram].strategy(),
                 lambda value: value[1][:1] != value[1][1:])
    assert shape[0].cod[-2:] == shape[1]


def test_HomogeneousMemory():
    x, m = map(feedback.Ty, "xm")
    f = feedback.Box('f', x @ (m @ m).delay(), x @ m @ m)
    assert HomogeneousMemory(f, m @ m)
    n = feedback.Ty('n')
    g = feedback.Box('g', x @ (m @ n).delay(), x @ m @ n)
    with raises(ValueError):
        HomogeneousMemory(g, m @ n)
    find(HomogeneousMemory[feedback.Diagram].strategy(),
         lambda value: True)


def test_Relabelling():
    x, y, z = cat.Ob('x'), cat.Ob('y'), cat.Ob('z')
    relabelling = Relabelling(((x, y), ))
    assert relabelling[x] == y and relabelling[z] == z
    assert list(relabelling) == [x] and len(relabelling) == 1
    assert bool(Relabelling())
    assert relabelling[cat.Box('f', x, x)] == cat.Box('f', y, y)
    rigid_x, rigid_y = rigid.Ty('x'), rigid.Ty('y')
    rotating = Relabelling(((rigid_x, rigid_y), ))
    functor = rigid.Functor(rotating, rotating)
    assert functor(rigid_x.l) == rigid_y.l and functor(rigid_x.r) == rigid_y.r
    rotated = rigid.Box('f', rigid_x.r, rigid_x @ rigid_x)
    assert rotating[rotated] == rigid.Box('f', rigid_y.r, rigid_y @ rigid_y)
    delayed = Relabelling(((feedback.Ty('u'), feedback.Ty('v')), ))
    delaying = feedback.Box('f', feedback.Ty('u').delay(), feedback.Ty('u'))
    assert delayed[delaying] == feedback.Box(
        'f', feedback.Ty('v').delay(), feedback.Ty('v'))


def test_Small():
    x = monoidal.Ty('x')
    assert Subsingleton(x).value == x
    with raises(ValueError):
        Subsingleton(x @ x)
    find(Subsingleton[monoidal.Ty].strategy(),
         lambda value: len(value.value) == 1)
    with raises(TypeError):
        resolve(int)


def test_BoundaryConnected():
    x = monoidal.Ty('x')
    f = monoidal.Box('f', x, x)
    scalar = monoidal.Box('s', monoidal.Ty(), monoidal.Ty())
    assert BoundaryConnected(f).value == f
    assert BoundaryConnected(f.to_hypergraph()).value
    assert BoundaryConnected(HorizontalPair(f, f)).value == (f, f)
    for value in (
            f @ scalar, scalar, scalar.to_map(), scalar.to_hypergraph()):
        with raises(ValueError):
            BoundaryConnected(value)
    find(BoundaryConnected[monoidal.Diagram].strategy(),
         lambda value: bool(value.value.boxes))
