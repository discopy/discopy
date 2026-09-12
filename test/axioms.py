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
    C1,
    Atomic,
    Axiom,
    AxiomFailure,
    BoundaryConnected,
    ComposablePair,
    ComposableTriple,
    Equation,
    FeedbackJoining,
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
    no_strategy,
    resolve,
    substitute,
)
from discopy.cat import Arrow, Box, Functor, Ob
from discopy.monoidal import Layer
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


def test_no_strategy():
    """ A class that does not generate its terms says so on `strategy`. """
    with raises(NotImplementedError) as err:
        Layer.strategy()
    assert "No search strategy implemented for Layer" in str(err.value)

    class Opted(Arrow):
        """ A class that would inherit a strategy for the wrong terms. """
        strategy = no_strategy

    with raises(NotImplementedError):
        Opted.strategy()
    assert Opted.axioms["unitality"] == Opted.unitality


def test_grid_states_its_law():
    """
    A grid is testable like any other: it states the composability its
    constructor enforces, and is checked against it once subscripted.
    """
    x, y = Ob('x'), Ob('y')
    composability = ComposablePair[Arrow].composability
    assert composability(ComposablePair(Box('f', x, y), Box('g', y, x)))
    drawn, = find(composability.strategy(), lambda _: True)
    assert composability(drawn)
    with raises(NoSuchExample):
        composability.falsify()

    with raises(NotImplementedError):
        ComposablePair.strategy()  # no factory to draw the cells from


def test_rules_are_inherited_and_bound():
    """ A category collects its rules through the MRO, like its axioms. """
    from discopy.axioms import Rule, search

    assert set(Arrow.rules) == {"identity", "box", "cut"}
    cut = Arrow.rules["cut"]
    assert isinstance(cut, Rule) and cut.category is Arrow
    assert repr(cut) == "cat.Arrow.cut"
    x, y = Ob('x'), Ob('y')
    arrow = find(
        search(Arrow, dom=x, cod=y, min_leaves=3, max_leaves=3,
               types=Ob.strategy()),
        lambda _: True)
    assert (arrow.dom, arrow.cod, len(arrow.inside)) == (x, y, 3)
    assert find(
        search(Arrow, dom=x, cod=x, max_leaves=0, types=Ob.strategy()),
        lambda _: True) == Arrow.id(x)


def test_leaf_applies_only_on_its_shape():
    """ A leaf is offered exactly when its pattern matches the sequent. """
    from discopy.axioms import Var, leaf, search

    A = Var.type('A')

    class Toy(Arrow):
        """ Loops on every object, and no cut. """
        cut = None

        @leaf((A, A))
        def loop(cls, dom, cod, A):
            return Box('loop', dom, cod)

    assert set(Toy.rules) == {"identity", "box", "loop"}
    x, y = Ob('x'), Ob('y')
    is_loop = lambda arrow: arrow.inside[0].name == 'loop'
    find(search(Toy, dom=x, cod=x, min_leaves=1, max_leaves=1,
                types=Ob.strategy()), is_loop)
    with raises(NoSuchExample):
        find(search(Toy, dom=x, cod=y, min_leaves=1, max_leaves=1,
                    types=Ob.strategy()), is_loop)


def test_rules_from_patterns():
    """ A pattern rule matches, builds, and hints at the sequent it needs. """
    from discopy import braided, rigid, traced

    x, y, z = map(rigid.Ty, "xyz")
    cupping = rigid.Diagram.rules["cupping"]
    assert cupping.applies(rigid.Diagram, x @ x.r, rigid.Ty(), 1)
    assert cupping.applies(rigid.Diagram, x.l @ x, rigid.Ty(), 1)
    assert not cupping.applies(rigid.Diagram, x @ y.r, rigid.Ty(), 1)
    assert not cupping.applies(rigid.Diagram, x @ x.r, rigid.Ty(), 2)
    types = rigid.Ty.strategy()
    assert find(cupping.shape(rigid.Diagram, x @ x.r, rigid.Ty(), types),
                lambda middle: not middle) == rigid.Ty()
    a, b, c = map(braided.Ty, "abc")
    braiding = braided.Diagram.rules["braiding"]
    hinted = find(braiding.shape(braided.Diagram, a @ b @ c, c, types),
                  lambda middle: middle == b @ a @ c)
    assert hinted == b @ a @ c
    tracing = traced.Diagram.rules["tracing_left"]
    assert tracing.applies(traced.Diagram, a, b, 2)
    assert not tracing.applies(traced.Diagram, a, b, 1)
    traced_arrow = find(
        traced.Diagram.strategy(dom=a, cod=b, min_leaves=2, max_leaves=2),
        lambda value: any(isinstance(box, traced.Trace) and box.left
                          for box in value.boxes))
    assert (traced_arrow.dom, traced_arrow.cod) == (a, b)


def test_Atomic():
    x, y = map(monoidal.Ty, "xy")
    assert Atomic(x).value == x
    with raises(ValueError):
        Atomic(x @ y)
    find(Atomic[monoidal.Ty].strategy(), lambda value: len(value.value) == 1)


def test_NonEmpty():
    x = monoidal.Ty('x')
    assert NonEmpty(x).value == x
    with raises(ValueError):
        NonEmpty(monoidal.Ty())
    find(NonEmpty[monoidal.Ty].strategy(), lambda value: len(value.value) > 1)


def test_HorizontalPair():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert HorizontalPair(f, g) == (f, g)
    with raises(ValueError):
        HorizontalPair(f)
    find(HorizontalPair[monoidal.Diagram].strategy(),
         lambda value: all(term.boxes for term in value))


def test_Square():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    assert Square(f, f, g, g) == (f, f, g, g)
    with raises(AxiomError):
        Square(f, f, f, f)
    find(Square[monoidal.Diagram].strategy(), lambda value: all(
        value[column].boxes or value[column + 2].boxes
        for column in range(2)))


def test_BoundaryConnected():
    x = monoidal.Ty('x')
    f = monoidal.Box('f', x, x)
    scalar = monoidal.Box('s', monoidal.Ty(), monoidal.Ty())
    assert BoundaryConnected(f).value == f
    assert BoundaryConnected(HorizontalPair(f, f)).value == (f, f)
    for value in (f @ scalar, scalar):
        with raises(ValueError):
            BoundaryConnected(value)
    find(BoundaryConnected[monoidal.Diagram].strategy(),
         lambda value: bool(value.value.boxes))


def test_TraceSuperposing():
    x, y, z = map(traced.Ty, "xyz")
    assert TraceSuperposing(traced.Id(x), y) == (traced.Id(x), y)
    with raises(AxiomError):
        TraceSuperposing(traced.Box('f', x, y), z)
    superposed = find(TraceSuperposing[traced.Diagram].strategy(),
                      lambda value: value[0].boxes and len(value[1]) > 1)
    assert superposed[0].boxes


def test_TraceNaturality():
    x, y = map(traced.Ty, "xy")
    f, g = traced.Box('f', x @ y, x @ x), traced.Box('g', x, y)
    assert TraceNaturalityLeft(f, x, g) == (f, x, g)
    with raises(ValueError):
        TraceNaturalityLeft(traced.Id(x @ y), x, traced.Id(x))
    h = traced.Box('h', y @ x, x @ x)
    assert TraceNaturalityRight(h, x, g) == (h, x, g)
    with raises(ValueError):
        TraceNaturalityRight(traced.Id(x @ y), x, traced.Id(y))
    for shape in (TraceNaturalityLeft, TraceNaturalityRight):
        find(shape[traced.Diagram].strategy(),
             lambda value: value[2].dom != value[2].cod)


def test_TraceDinaturality():
    x, y, z = map(traced.Ty, "xyz")
    f, g = traced.Box('f', x @ z, y @ z), traced.Box('g', y, x)
    assert TraceDinaturalityLeft(f, g) == (f, g)
    with raises(ValueError):
        TraceDinaturalityLeft(g, f)
    h = traced.Box('h', z @ x, z @ y)
    assert TraceDinaturalityRight(h, g) == (h, g)
    with raises(ValueError):
        TraceDinaturalityRight(g, h)
    shape = find(TraceDinaturalityRight[traced.Diagram].strategy(),
                 lambda value: value[1].dom != value[1].cod)
    sliding = shape[1]
    assert shape[0].dom[-len(sliding.cod):] == sliding.cod
    assert shape[0].cod[-len(sliding.dom):] == sliding.dom


def test_Currying():
    x, y, z = map(biclosed.Ty, "xyz")
    f, g = biclosed.Box('f', z @ y, x), biclosed.Box('g', y @ z, x)
    assert LeftCurrying(f, x, y) == (f, x, y)
    assert RightCurrying(g, x, y) == (g, x, y)
    with raises(ValueError):
        LeftCurrying(g, x, y)
    with raises(ValueError):
        RightCurrying(f, x, y)
    for shape in (LeftCurrying, RightCurrying):
        drawn = find(shape[biclosed.Diagram].strategy(), lambda value:
                     len(value[0].dom) > 1 and not value[0].dom.is_exp)
        assert drawn[0].cod == drawn[1]


def test_FeedbackJoining():
    x, y, z = map(feedback.Ty, "xyz")
    memory = y @ z
    g = feedback.Box('g', x @ memory.delay(), x @ memory)
    assert FeedbackJoining(g, memory) == (g, memory)
    with raises(ValueError):
        FeedbackJoining(g, feedback.Ty())
    with raises(ValueError):
        FeedbackJoining(feedback.Box('h', x @ memory, x @ memory), memory)
    shape = find(FeedbackJoining[feedback.Diagram].strategy(),
                 lambda value: value[1][:1] != value[1][1:])
    assert shape[0].cod[-2:] == shape[1]


def test_pattern_matching():
    from discopy import rigid
    from discopy.axioms import Var, Pattern

    X, Y = Var.atom('X'), Var.atom('Y')
    A, B = Var.type('A'), Var.type('B')
    a, b, c = map(rigid.Ty, "abc")
    assert isinstance(X @ X.r, Pattern) and (X @ X.r).fixed
    assert not (A @ X).fixed
    assert list((X @ X.r).match(a @ a.r)) == [{"X": a}]
    assert list((X @ X.r).match(a @ b.r)) == []
    assert list((X @ X.l).match(a.r @ a)) == [{"X": a.r}]
    assert list((X.r @ X).match(a.r @ a)) == [{"X": a}]
    assert list((A @ X @ B).match(a @ b @ c)) == [
        {"A": rigid.Ty(), "X": a, "B": b @ c},
        {"A": a, "X": b, "B": c},
        {"A": a @ b, "X": c, "B": rigid.Ty()}]
    assert list((A @ A).match(a @ b @ a @ b)) == [{"A": a @ b}]
    assert list(Pattern().match(rigid.Ty())) == [{}]
    assert list((X @ Y).match(a)) == []
    x = cat.Ob('x')
    assert list(Pattern((A, )).match(x)) == [{"A": x}]


def test_pattern_exponentials_and_delays():
    from discopy import biclosed, feedback
    from discopy.axioms import Var

    B, E = Var.atom('B'), Var.atom('E')
    x, y = map(biclosed.Ty, "xy")
    assert list(((B << E) @ E).match((y << x) @ x)) == [{"B": y, "E": x}]
    assert list(((B << E) @ E).match((y << x) @ y)) == []
    assert list((E @ (E >> B)).match(x @ (x >> y))) == [{"E": x, "B": y}]
    M, A = Var.pair('M'), Var.type('A')
    u, v, w = map(feedback.Ty, "uvw")
    assert list((A @ M.delay()).match(w @ (u @ v).delay())) == [
        {"A": w, "M": u @ v}]
    assert list((A @ M).match(w @ u @ v)) == [{"A": w, "M": u @ v}]


def test_pattern_instantiation():
    from discopy import rigid
    from discopy.axioms import Var, Pattern

    X, A = Var.atom('X'), Var.type('A')
    a, b = map(rigid.Ty, "ab")
    assert (A @ X @ X.r).instantiate({"A": a @ b, "X": a}) == a @ b @ a @ a.r
    assert Pattern().instantiate({}, unit=rigid.Ty()) == rigid.Ty()
    drawn = find(
        (X @ X.r).strategy(rigid.Diagram), lambda value: len(value) == 2)
    assert drawn[1:] == drawn[:1].r
    assert find((A @ X).strategy(rigid.Diagram, {"X": a}),
                lambda value: len(value) == 3)[2:] == a
