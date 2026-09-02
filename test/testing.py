""" DisCoPy's property-testing module in action, see PROPTEST.md. """

from __future__ import annotations

from hypothesis import find
from hypothesis import strategies as st
from pytest import raises

from discopy import cat, closed, feedback, rigid, symmetric, testing
from discopy.abc import Equation as AbstractEquation
from discopy.cat import Arrow, Box, Equation, Functor, Ob
from discopy.testing import (
    C0, C1, Atomic, Axiom, AxiomFailure, BoundaryConnected, ComposablePair,
    FeedbackJoining, FeedbackVanishing, HomogeneousMemory, HorizontalPair,
    LeftCurrying, Natural, NonEmpty, Relabelled, Relabelling, RightCurrying,
    Small, Strategy, TraceDinaturalityLeft, TraceDinaturalityRight,
    TraceNaturalityLeft, TraceNaturalityRight, TraceSuperposing,
    assert_axioms, assert_strategy_finds, axiom, resolve)
from discopy.utils import AxiomError, factory


class TypeStrategy(Strategy):
    """ Generate a type of length at most two over three atomic names. """

    @classmethod
    def strategy(cls, *, min_length=0):
        return st.lists(
            st.sampled_from("uvw"),
            min_size=min_length, max_size=2).map(lambda names: cls(*names))


class BoxStrategy(Strategy):
    """ Generate a single generator box with the requested boundary. """

    box_factory: type

    @classmethod
    def strategy(cls, *, dom=None, cod=None, **params):
        doms = cls.ob.strategy() if dom is None else st.just(dom)
        cods = cls.ob.strategy() if cod is None else st.just(cod)
        return st.tuples(doms, cods).map(
            lambda pair: cls.box_factory("f", *pair))


@factory
class TracedTy(TypeStrategy, symmetric.Ty):
    """ The objects of a toy traced category. """


@factory
class TracedDiagram(BoxStrategy, symmetric.Diagram):
    """ The arrows of a toy traced category. """
    ob = TracedTy


class TracedBox(symmetric.Box, TracedDiagram):
    """ A generator of the toy traced category. """


@factory
class ClosedTy(TypeStrategy, closed.Ty):
    """ The objects of a toy closed category. """


@factory
class ClosedDiagram(closed.Diagram):
    """ The arrows of a toy closed category. """
    ob = ClosedTy


@factory
class FeedbackTy(TypeStrategy, feedback.Ty):
    """ The objects of a toy feedback category. """


@factory
class FeedbackDiagram(BoxStrategy, feedback.Diagram):
    """ The arrows of a toy feedback category. """
    ob = FeedbackTy


class FeedbackBox(feedback.Box, FeedbackDiagram):
    """ A generator of the toy feedback category. """


TracedDiagram.box_factory = TracedBox
FeedbackDiagram.box_factory = FeedbackBox


def test_axioms():
    assert_axioms(Arrow, Functor)


def test_strategy():
    assert_strategy_finds(TracedDiagram, TracedBox)
    x, y = Ob('x'), Ob('y')
    find(Ob.strategy(), lambda ob: ob.name == "a")
    assert find(Arrow.strategy(dom=x, cod=x), lambda _: True) == Arrow.id(x)
    assert find(Arrow.strategy(dom=x, cod=y), lambda _: True).cod == y
    assert find(Arrow.strategy(dom=x), lambda _: True).dom == x
    assert find(Arrow.strategy(cod=y), lambda _: True).cod == y
    assert find(Box.strategy(dom=x), lambda _: True).dom == x


def test_natural():
    assert Natural() == 0 and Natural(2) @ Natural(3) == Natural(5)
    assert len(Natural(3)) == 3
    assert Natural(1).__matmul__("x") is NotImplemented
    with raises(ValueError):
        Natural(-1)
    assert repr(Natural(2)) == "testing.Natural(2)"
    assert eval(repr(Natural(2))) == testing.Natural(2)
    assert Natural.equation_factory(Natural(1), Natural(1))
    assert find(Natural.strategy(), lambda number: number == 1) == 1


def test_argument_wrappers():
    one, two = Natural(1), Natural(2)
    assert Atomic(one).value == NonEmpty(one).value == Small(one).value == one
    for wrapper, value in (
            (Atomic, two), (NonEmpty, Natural()), (Small, two)):
        with raises(ValueError):
            wrapper(value)
    assert find(
        Atomic[Natural].strategy(), lambda _: True) == Atomic[Natural](one)
    assert find(NonEmpty[Natural].strategy(), lambda _: True).value
    assert len(
        find(Small[Natural].strategy(), lambda _: True).value) <= 1
    pair = find(
        resolve(NonEmpty[ComposablePair[TracedDiagram]]), lambda _: True)
    assert isinstance(pair.value, ComposablePair)
    with raises(TypeError):
        resolve(int)


def test_boundary_connected():
    x = symmetric.Ty('x')
    box = symmetric.Box('f', x, x)
    scalar = symmetric.Box('s', symmetric.Ty(), symmetric.Ty())
    assert BoundaryConnected(box.to_hypergraph()).value
    assert BoundaryConnected((box, box)).value == (box, box)
    for value in (scalar, scalar.to_map(), scalar.to_hypergraph()):
        with raises(ValueError):
            BoundaryConnected(value)

    class Terms(Strategy):
        """ A stub carrier quantifying over one connected box. """
        @classmethod
        def strategy(cls, *, boundary_connected=False):
            return st.just(box)

    assert find(
        BoundaryConnected[Terms].strategy(), lambda _: True).value == box


def test_pasting_diagram():
    x, y = TracedTy('u'), TracedTy('v')
    box = TracedBox('f', x, y)
    assert HorizontalPair(box, box) == (box, box)
    with raises(ValueError):
        HorizontalPair(box)
    with raises(AxiomError):
        ComposablePair(box, box)


def test_trace_wrappers():
    x, y = TracedTy('u'), TracedTy('v')
    identity = TracedDiagram.id(x)
    assert TraceSuperposing(identity, y) == (identity, y)
    for wrapper in (TraceSuperposing, TraceNaturalityLeft,
                    TraceNaturalityRight, TraceDinaturalityLeft,
                    TraceDinaturalityRight):
        find(wrapper[TracedDiagram].strategy(), lambda _: True)
    with raises(ValueError):
        TraceNaturalityLeft(TracedBox('f', x, x), y, TracedBox('g', x, y))
    with raises(ValueError):
        TraceDinaturalityLeft(TracedBox('f', x, x), TracedBox('g', y, y))


def test_currying_wrappers():
    base, exponent = ClosedTy('u'), ClosedTy('v')
    evaluation = ClosedDiagram.ev(base, exponent, left=True)
    assert LeftCurrying(evaluation, base, exponent)\
        == (evaluation, base, exponent)
    with raises(ValueError):
        RightCurrying(evaluation, base, exponent)
    find(LeftCurrying[ClosedDiagram].strategy(), lambda _: True)
    find(RightCurrying[ClosedDiagram].strategy(), lambda _: True)


def test_feedback_wrappers():
    x, y = FeedbackTy('u'), FeedbackTy('v')
    box, memory = FeedbackBox('f', x, y), x @ y
    assert FeedbackVanishing(box, FeedbackTy()) == (box, FeedbackTy())
    with raises(ValueError):
        FeedbackVanishing(box, x)
    joining = FeedbackBox('f', x @ memory.delay(), x @ memory)
    assert FeedbackJoining(joining, memory) == (joining, memory)
    with raises(ValueError):
        FeedbackJoining(joining, x)
    with raises(ValueError):
        FeedbackJoining(FeedbackBox('f', x @ memory, x @ memory), memory)
    with raises(ValueError):
        FeedbackJoining(FeedbackBox(
            'f', x @ memory.delay(), x @ memory.delay()), memory)
    assert HomogeneousMemory(
        FeedbackBox('f', x @ (x @ x).delay(), x @ x @ x), x @ x)
    with raises(ValueError):
        HomogeneousMemory(joining, memory)
    for wrapper in (FeedbackVanishing, FeedbackJoining, HomogeneousMemory):
        find(wrapper[FeedbackDiagram].strategy(), lambda _: True)


def test_relabelling():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    relabelling = Relabelling(((x, y), ))
    assert relabelling[x] == y and relabelling[z] == z
    assert list(relabelling) == [x] and len(relabelling) == 1
    assert bool(Relabelling()) and relabelling.send(x) == y
    rigid_x, rigid_y = rigid.Ty('x'), rigid.Ty('y')
    rotating = Relabelling(((rigid_x, rigid_y), ))
    assert rotating[rigid_x.l] == rigid_y.l
    assert rotating[rigid_x.r] == rigid_y.r
    assert rotating.send(rigid_x @ rigid_x.l) == rigid_y @ rigid_y.l
    delayed = Relabelling(((FeedbackTy('u'), FeedbackTy('v')), ))
    assert delayed[FeedbackTy('u').delay()] == FeedbackTy('v').delay()
    relabelled = Relabelled(relabelling)
    assert relabelled[Box('f', x, x)] == Box('f', y, y)
    assert list(relabelled) == [] and not len(relabelled) and bool(relabelled)


def test_monoid_axioms():
    x, y, z = TracedTy('u'), TracedTy('v'), TracedTy('w')
    assert TracedTy.monoid_unitality(x)
    assert TracedTy.monoid_associativity((x, y, z))


def test_axiom_binding():
    assert repr(Axiom(lambda cls: NotImplemented)) == "Axiom(<lambda>)"
    assert eval(repr(Arrow.unitality)) == cat.Arrow.unitality
    assert hash(Arrow.unitality) == hash(eval(repr(Arrow.unitality)))
    assert Arrow.unitality != Functor.unitality
    assert Functor.dagger_involution() is NotImplemented
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented)()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).falsify()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).strategy()
    assert Axiom(classmethod(lambda cls: NotImplemented)).bind(Arrow)()\
        is NotImplemented
    box = Box('f', Ob('x'), Ob('y'))
    assert Arrow.unitality(box)
    broken = Arrow.unitality.weaken(f=Atomic[C1]).failing("Never holds.")
    assert broken.subspaces == {"f": Atomic[C1]}
    with raises(AxiomFailure) as failure:
        broken(Atomic(box))
    assert failure.value.equation


def test_inapplicable():
    law = Arrow.unitality.inapplicable("No identities to cancel.")
    assert law.name == "unitality"
    assert law.__doc__ == "No identities to cancel."
    assert law() is NotImplemented


def test_modulo():
    law = Arrow.unitality.modulo(lambda term: term.dom).bind(Arrow)
    assert law(Box('f', Ob('x'), Ob('y')))


def test_weaken():
    for subspace in (Atomic[C1], Atomic[TracedTy]):
        law = TracedTy.monoid_unitality.weaken(x=subspace).bind(TracedTy)
        assert law.modulo(lambda term: term).subspaces == law.subspaces
        args = find(law.strategy(), lambda _: True)
        assert isinstance(args[0], Atomic) and law(*args)


def test_element_law():
    @axiom
    def preserves_identity(self, x: C0) -> AbstractEquation:
        """ A functor preserves the identity on each object. """
        return Equation(self(Arrow.id(x)), Arrow.id(self(x)))

    law = preserves_identity.bind(Functor)
    assert law.is_method
    args = find(law.strategy(), lambda _: True)
    assert law(*args)


def test_falsify():
    counterexample, = Functor.unitality.falsify()
    assert isinstance(counterexample, Functor)


def test_assert_axioms_refusal():
    def refuse(cls, f: C1) -> AbstractEquation:
        """ The equation never builds its terms. """
        raise AxiomError

    class Refusing(Arrow):
        """ A carrier whose extra law refuses to build its terms. """

    Refusing.refuse = Axiom(refuse).failing("The equation never builds.")
    assert_axioms(Refusing)
    assert Refusing.refuse.falsify()


def test_axioms_of_carrier():
    axioms = Functor.axioms
    assert "unitality" in axioms and axioms["unitality"].broken

    class Hidden(Arrow):
        """ Assigning a non-axiom over an inherited law drops it. """
        unitality = None

    assert "unitality" not in Hidden.axioms
