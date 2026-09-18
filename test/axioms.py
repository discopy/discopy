""" DisCoPy's property-testing module in action. """

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
    Testable,
    assert_axioms,
    axiom,
    no_strategy,
    resolve,
    substitute,
)
from discopy.cat import Arrow, Box, Functor, Ob
from discopy.monoidal import Diagram
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
        Diagram.strategy()
    assert "No search strategy implemented for Diagram" in str(err.value)

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
