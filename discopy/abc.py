"""
The abstract base classes for categories.

These mirror the concrete hierarchy of :mod:`discopy` modules: each class adds
the characteristic generator of its categorical structure as an
:func:`abc.abstractmethod`, e.g. :class:`BraidedCategory` is a
:class:`MonoidalCategory` with an abstract :meth:`BraidedCategory.braid`.

.. raw:: html
    :file: api/architecture.html

Software dependencies between modules go top-to-bottom, left-to-right and
forgetful functors between categories go the other way.

Each class also declares its :func:`discopy.axioms.axiom` equations, which
every free category inherits along with the structure they axiomatise:
:class:`Category` states the unitality and associativity of composition,
the typing of its identities and composites, and the involution and
contravariance of its dagger; a :class:`ColouredMonoid` inherits them as
the unitality and associativity of its product, its composition.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Category
    ColouredMonoid
    Monoid
    Nat
    MonoidalCategory
    PRO
    TracedCategory
    ResiduatedMonoid
    BiclosedCategory
    Pregroup
    RigidCategory
    PivotalCategory
    BraidedCategory
    PROB
    SymmetricCategory
    PROP
    MarkovCategory
    ClosedCategory
    FeedbackCategory
    BalancedCategory
    RibbonCategory
    CompactCategory
    HypergraphCategory
    NamedGeneric
"""

from __future__ import annotations
from types import NoneType

from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import permutations
from dataclasses import dataclass, replace
from typing import ClassVar

from discopy.axioms import (  # noqa: F401
    C0, C1, C2, Atom, Axiom, Bool, Count, Equation, NonEmpty, Pair, Rule,
    Serialisable, Testable, Var, axiom, generator, inapplicable, rule)
from discopy.utils import (  # noqa: F401
    NamedGeneric, classproperty, factory_name, unbiased)


class Category[C0, C1: Category](Testable, ABC):
    """
    A category is a class with two class variables ``ob, ar``, two attributes
    ``dom, cod`` and two methods ``id, then``.

    This base class also implements syntactic sugar :code:`>>` and :code:`<<`
    for forward and backward composition with the method :code:`then`.

    Example
    -------
    >>> class List(list, Category):
    ...     ob, dom, cod = type(None), None, None
    ...     def then(self, other):
    ...         return self + other
    >>> assert List([1, 2]) >> List([3]) == List([1, 2, 3])
    >>> assert List([3]) << List([1, 2]) == List([1, 2, 3])
    """
    ob: ClassVar[type[C0]]
    factory: ClassVar[type[C1]]
    dom: C0
    cod: C0

    #: Backward-compatible alias for :attr:`factory`, since types are
    #: themselves the objects of diagrams.
    ar = classproperty(lambda cls: getattr(cls, "factory", cls))

    @classmethod
    def equation_factory(cls, *terms) -> Equation:
        """
        Construct an equation, using strict equality by default.

        A class that quotients its equations overrides this, e.g. by
        hypergraph isomorphism from symmetric categories on, so an axiom
        built with it is checked up to whatever quotient the category
        defines — and :meth:`discopy.axioms.Axiom.modulo` weakens it
        further.
        """
        return Equation(*terms)

    @classmethod
    @rule(boxes=0)
    @abstractmethod
    def id[A: C0](cls, dom: A) -> C1[A, A]:
        """
        Identity morphism on an object :code:`dom: C0`, to be instantiated:
        as a rule, ``x ⊢ x`` with no box is the identity.

        Parameters:
            dom (C0) : The domain of an identity is also its codomain.
        """

    @rule(boxes=0)
    @abstractmethod
    @unbiased
    def then[A: C0, B: C0, C: C0](
            self: C1[A, B], other: C1[B, C]) -> C1[A, C]:
        """
        Sequential composition of `n >= 1` morphisms, to be instantiated:
        as a rule, ``x ⊢ z`` is the composition of ``x ⊢ y`` and ``y ⊢
        z`` at a middle ``y`` drawn from the types or from a boundary
        some rule :func:`discopy.axioms.hints` at.

        Parameters:
            other : The other morphism to compose sequentially.
        """

    def is_composable(self, other: C1) -> bool:
        """
        Whether two morphisms are composable, i.e. the codomain of the first is
        the domain of the second.

        Parameters:
            other : The other morphism.
        """
        return self.cod == other.dom

    def is_parallel(self, other: Category) -> bool:
        """
        Whether two morphisms are parallel, i.e. they have the same
        domain and codomain.

        Parameters:
            other : The other morphism.
        """
        return (self.dom, self.cod) == (other.dom, other.cod)

    @classproperty
    def rules(cls) -> dict[str, Rule]:
        """
        The inference rules inherited by ``cls``: the rule each structural
        method carries, see :func:`discopy.axioms.generator` and
        :func:`discopy.axioms.rule`, collected through the MRO as
        :attr:`axioms` are, so a category generates exactly the structures
        whose laws it must satisfy, each bound to ``cls`` and owned by the
        class declaring it, whose levels its annotations name. A plain
        override of a method keeps the rule of the method it overrides,
        one decorated with :func:`discopy.axioms.inapplicable` drops it.
        """
        found = {}
        for base in reversed(cls.__mro__):
            for name, value in base.__dict__.items():
                carried = Rule.of(value)
                if carried is not None:
                    found[name] = replace(
                        carried, category=cls, owner=carried.owner or base)
        return found

    @classproperty
    def generators(cls) -> dict[str, Rule]:
        """
        The generators the search may invoke: the rules of :attr:`rules`
        that build a box from no premise — a free box, the structural
        boxes of the level, its permutations — the inapplicable ones left
        out, collected like :attr:`axioms`. A class adjusts the set by
        assigning a dictionary
        of rules instead, :meth:`discopy.axioms.Rule.constant` giving the
        rule of one given box, so that its strategy draws from a fixed
        vocabulary, e.g. the words of a pregroup grammar or the gates of
        a circuit, while the structure of its :attr:`rules` stays.

        A rule with premises stays invoked, so one whose premise the
        vocabulary cannot derive, a trace, is declared
        :func:`discopy.axioms.inapplicable`, or every derivation reaching
        it is rejected.

        >>> from hypothesis import find
        >>> from discopy.axioms import Rule
        >>> from discopy.grammar import pregroup
        >>> n, s = pregroup.Ty('n'), pregroup.Ty('s')
        >>> Alice, sleeps = pregroup.Word('Alice', n), pregroup.Word(
        ...     'sleeps', n.r @ s)
        >>> class Sentence(pregroup.Diagram):
        ...     generators = {
        ...         "cups": pregroup.Diagram.generators["cups"],
        ...         **{w.name: Rule.constant(w) for w in (Alice, sleeps)}}
        >>> print(find(Sentence.strategy(min_leaves=3), bool).foliation())
        Alice @ sleeps >> Cup(n, n.r) @ s
        """
        return {
            name: rule for name, rule in cls.rules.items()
            if rule.boxes and not rule.premises and rule.reason is None}

    @rule(applies=lambda cls, dom, cod, size: size == 1)
    def box(cls, draw, dom, cod, size, types):
        """ ``x ⊢ y`` with one box is a fresh generator. """
        from hypothesis import strategies as st

        name = str(draw(st.uuids()))
        return [], lambda: cls.box_factory(name, dom, cod)

    @axiom
    def unitality(
            cls, f: C1) -> Equation[C1]:
        """ Left and right unitality of composition. """
        return cls.equation_factory(
            cls.id(f.dom).then(f), f, f.then(cls.id(f.cod)))

    @axiom
    def associativity[A: C0, B: C0, C: C0, D: C0](
            cls, f: C1[A, B], g: C1[B, C], h: C1[C, D]) -> Equation[C1]:
        """ Associativity of composition. """
        return cls.equation_factory(
            f.then(g).then(h), f.then(g.then(h)))

    @axiom
    def identity_typing(
            cls, x: C0) -> Equation[C0]:
        """ Typing of identity morphisms. """
        identity = cls.id(x)
        return cls.ob.equation_factory(identity.dom, x, identity.cod)

    @axiom
    def composition_dom_typing[A: C0, B: C0, C: C0](
            cls, f: C1[A, B], g: C1[B, C]) -> Equation[C0]:
        """ Domain typing of composition. """
        return cls.ob.equation_factory(f.then(g).dom, f.dom)

    @axiom
    def composition_cod_typing[A: C0, B: C0, C: C0](
            cls, f: C1[A, B], g: C1[B, C]) -> Equation[C0]:
        """ Codomain typing of composition. """
        return cls.ob.equation_factory(f.then(g).cod, g.cod)

    @axiom
    def dagger_involution(
            cls, f: C1) -> Equation[C1]:
        """ The dagger is involutive. """
        return cls.equation_factory(f.dagger().dagger(), f)

    @axiom
    def dagger_contravariance[A: C0, B: C0, C: C0](
            cls, f: C1[A, B], g: C1[B, C]) -> Equation[C1]:
        """ The dagger reverses composition. """
        return cls.equation_factory(
            f.then(g).dagger(), g.dagger().then(f.dagger()))

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


class ColouredMonoid[C0, C1: ColouredMonoid](Category[C0, C1]):
    """
    A coloured monoid is a category whose sequential composition ``then`` is
    given by a monoidal ``tensor``, with the objects ``C0`` (its colours) as
    the boundaries of its morphisms.

    An ordinary :obj:`Monoid` is the special case with a single, trivial
    colour, i.e. :class:`type(None)`. We do not enforce this so
    that e.g. :class:`monoidal.Ty` can take colours as objects.
    """
    @classmethod
    def id(cls, dom: C0 = None) -> C1:
        """The monoidal unit, i.e. the empty tensor ``cls()``."""
        return cls()

    @classmethod
    def unit(cls, colour: C0 = None) -> C0 | C1:
        """
        The unit at a colour, i.e. the identity on it.

        It need not be an element of the monoid, which is why it may land in
        ``C0``: the layers of :class:`monoidal.Layer` are closed under
        ``tensor`` but the empty one is a type rather than a layer.
        """
        return cls.id(colour)

    @abstractmethod
    def tensor(self, *objects: C1) -> C1:
        """ The n-ary product of a monoid for ``n > 0``. """

    def then(self, *others: C1) -> C1:
        """Sequential composition, given by the monoid product."""
        return self.tensor(*others)

    @classmethod
    def cast(cls, atoms) -> C1:
        """
        The element of a tuple of atoms, or of a single atom; an element of
        the monoid is unchanged.

        Parameters:
            atoms : An element, a tuple of atoms or a single atom.

        >>> assert Nat.cast(2) == Nat(2) == Nat.cast(Nat(2))
        """
        if isinstance(atoms, cls):
            return atoms
        return cls(*atoms) if isinstance(atoms, tuple) else cls(atoms)

    @classmethod
    def whisker(cls, other: C0 | C1) -> C1:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return other if isinstance(other, cls) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(other)

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class Monoid[C1: Monoid](ColouredMonoid[NoneType, C1]):
    """ A monoid is a coloured monoid with a single, trivial colour. """


@dataclass
class Nat(Monoid["Nat"]):
    """
    ``Nat`` is the free monoid on one generator, i.e. the natural numbers
    with addition as tensor. It is also a sequence over its unary encoding:
    :meth:`__len__` gives back the natural number itself and slicing reads
    it off as a sequence of ``1``'s, e.g. ``Nat(3)[:1] == Nat(1)``.

    Parameters:
        n : The natural number.
    """
    n: int = 0

    def tensor(self, *others: Nat) -> Nat:
        if any(not isinstance(other, Nat) for other in others):
            return NotImplemented  # This allows whiskering on the left.
        return type(self)(self.n + sum(other.n for other in others))

    def __len__(self) -> int:
        return self.n

    def __index__(self) -> int:
        return self.n

    def __str__(self) -> str:
        return str(self.n)

    def __getitem__(self, key: int | slice) -> Nat:
        """
        Slicing a natural number reads it off as a sequence of ``1``'s.

        Parameters:
            key : An integer or a slice.
        """
        if isinstance(key, slice):
            return type(self)(len(range(self.n)[key]))
        if key >= self.n or key < -self.n:
            raise IndexError
        return type(self)(1)


class Colour(Testable):
    """
    The 0-cells of a :class:`TwoCategory`, the colours of the regions of
    a diagram: the objects of the :class:`ColouredMonoid` of its 1-cells,
    :class:`discopy.monoidal.Colour` for the diagrams of DisCoPy. A class
    of colours implements :meth:`strategy` to be drawn as the colours of a
    law of 2-categories.
    """


class TwoCategory[C0: Colour, C1: ColouredMonoid, C2: TwoCategory](
        Category[C1, C2]):
    """
    A 2-category is a :class:`Category` whose objects are the elements
    of a :class:`ColouredMonoid` — 1-cells between the colours ``C0``,
    its 0-cells — and whose arrows are 2-cells between parallel 1-cells,
    with a :meth:`tensor` composing them horizontally along a colour: a
    monoidal category is the case of a single colour, its wires the
    1-cells and its boxes the 2-cells, see :class:`MonoidalCategory`.
    """
    @rule(boxes=0)
    @abstractmethod
    @unbiased
    def tensor[
            X: C0, Y: C0, Z: C0, A: C1[X, Y], B: C1[X, Y],
            C: C1[Y, Z], D: C1[Y, Z]](
            self: C2[A, B], other: C2[C, D]) -> C2[A @ C, B @ D]:
        """
        Parallel composition of ``n >= 0`` morphisms, to be instantiated:
        as a rule, ``a @ c ⊢ b @ d`` is the tensor of ``a ⊢ b`` and ``c ⊢
        d``, one of them an identity when it whiskers the other, split
        where the colours meet.

        Parameters:
            other : The other morphism to compose in parallel.
        """

    @axiom
    def bifunctoriality[
            X: C0, Y: C0, Z: C0, A: C1[X, Y], B: C1[X, Y], U: C1[X, Y],
            C: C1[Y, Z], D: C1[Y, Z], V: C1[Y, Z]](
            cls, f: C2[A, B], g: C2[C, D], h: C2[B, U], k: C2[D, V]
            ) -> Equation[C2[A @ C, U @ V]]:
        """ Bifunctoriality of the tensor. """
        return cls.equation_factory(
            f @ g >> h @ k, (f >> h) @ (g >> k))

    @axiom
    def tensor_unitality[X: C0, Y: C0, Z: C0, A: C1[X, Y], B: C1[Y, Z]](
            cls, a: A, b: B) -> Equation[C2[A @ B, A @ B]]:
        """ Preservation of identities by tensor. """
        return cls.equation_factory(
            cls.id(a) @ cls.id(b), cls.id(a @ b))

    @axiom
    def tensor_dom_typing[
            X: C0, Y: C0, Z: C0, A: C1[X, Y], B: C1[X, Y],
            C: C1[Y, Z], D: C1[Y, Z]](
            cls, f: C2[A, B], g: C2[C, D]) -> Equation[C1[X, Z]]:
        """ Domain typing of tensor. """
        return cls.ob.equation_factory((f @ g).dom, f.dom @ g.dom)

    @axiom
    def tensor_cod_typing[
            X: C0, Y: C0, Z: C0, A: C1[X, Y], B: C1[X, Y],
            C: C1[Y, Z], D: C1[Y, Z]](
            cls, f: C2[A, B], g: C2[C, D]) -> Equation[C1[X, Z]]:
        """ Codomain typing of tensor. """
        return cls.ob.equation_factory((f @ g).cod, f.cod @ g.cod)

    @axiom
    def dagger_monoidality[
            X: C0, Y: C0, Z: C0, A: C1[X, Y], B: C1[X, Y],
            C: C1[Y, Z], D: C1[Y, Z]](
            cls, f: C2[A, B], g: C2[C, D]) -> Equation[C2[B @ D, A @ C]]:
        """ The dagger distributes over the tensor. """
        return cls.equation_factory(
            (f @ g).dagger(), f.dagger() @ g.dagger())


class MonoidalCategory[C0: ColouredMonoid, C1: MonoidalCategory](
        TwoCategory[Colour, C0, C1]):
    """
    A monoidal category is a :class:`TwoCategory` whose 0-cells are the
    colours of its objects, a coloured monoid, see :class:`Colour`, and whose
    ``tensor`` composes both its objects and its morphisms: it inherits
    the rules and the axioms of the tensor, its wires being the 1-cells
    and its boxes the 2-cells. A subclass whose wires cross or bend
    declares the one trivial colour instead, see :class:`BraidedCategory`.

    This base class also implements syntactic sugar :code:`@` for whiskering.
    """
    @classmethod
    def whisker(cls, other: C0 | C1) -> C1:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return other if isinstance(other, MonoidalCategory) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(self.whisker(other))

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class PRO[C1: PRO](MonoidalCategory[Nat, C1]):
    """
    A PRO is a :class:`MonoidalCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free monoidal category on one generator.
    """


class TracedCategory[C0, C1](MonoidalCategory[C0, C1]):
    """
    A traced category is a :class:`MonoidalCategory` with a method
    :code:`trace` for the partial trace of a morphism over some objects.
    """
    @rule
    @abstractmethod
    def trace[A: C0, B: C0, M: Atom[C0], L: Bool](
            self: C1[L[M @ A, A @ M], L[M @ B, B @ M]],
            n: int = 1, left: L = False) -> C1[A, B]:
        """
        The trace of a morphism, to be instantiated.

        Tracing no object at all is the identity, i.e. the vanishing axiom
        ``f.trace(0) == f``, see `nLab
        <https://ncatlab.org/nlab/show/traced+monoidal+category>`_. As a
        rule, ``x ⊢ y`` is the trace of ``m @ x ⊢ m @ y`` or of ``x @ m ⊢
        y @ m`` over an atom, on the left or on the right.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """

    @axiom
    def trace_vanishing(
            cls, f: C1) -> Equation[C1]:
        """ Vanishing of a trace over the unit. """
        return cls.equation_factory(
            f.trace(0), f, f.trace(0, left=True))

    @axiom
    def trace_superposing_left[X: Atom[C0], A: C0, B: C0](
            cls, f: C1[X @ A, X @ B], obj: C0) -> Equation[C1]:
        """ Left-oriented superposing. """
        return cls.equation_factory(
            (f @ obj).trace(left=True), f.trace(left=True) @ obj)

    @axiom
    def trace_superposing_right[X: Atom[C0], A: C0, B: C0](
            cls, f: C1[A @ X, B @ X], obj: C0) -> Equation[C1]:
        """ Right-oriented superposing. """
        return cls.equation_factory(
            (obj @ f).trace(), obj @ f.trace())

    @axiom
    def trace_naturality_left[N: NonEmpty[C0], A: C0, B: C0](
            cls, f: C1[N @ A, N @ B], x: N, g: C1[B, A]) -> Equation[C1]:
        """ Left-oriented trace naturality. """
        return cls.equation_factory(
            (x @ g).then(f).then(x @ g).trace(len(x), left=True),
            g.then(f.trace(len(x), left=True)).then(g))

    @axiom
    def trace_naturality_right[N: NonEmpty[C0], A: C0, B: C0](
            cls, f: C1[A @ N, B @ N], x: N, g: C1[B, A]) -> Equation[C1]:
        """ Right-oriented trace naturality. """
        return cls.equation_factory(
            (g @ x).then(f).then(g @ x).trace(len(x)),
            g.then(f.trace(len(x))).then(g))

    @axiom
    def trace_dinaturality_left[
            N: NonEmpty[C0], K: NonEmpty[C0], A: C0, B: C0](
            cls, f: C1[N @ A, K @ B], g: C1[K, N]) -> Equation[C1]:
        """ Left-oriented trace dinaturality. """
        source, target = g.cod, g.dom
        base, cobase = f.dom[len(source):], f.cod[len(target):]
        return cls.equation_factory(
            f.then(g @ cobase).trace(len(source), left=True),
            (g @ base).then(f).trace(len(target), left=True))

    @axiom
    def trace_dinaturality_right[
            N: NonEmpty[C0], K: NonEmpty[C0], A: C0, B: C0](
            cls, f: C1[A @ N, B @ K], g: C1[K, N]) -> Equation[C1]:
        """ Right-oriented trace dinaturality. """
        source, target = g.cod, g.dom
        base = f.dom[:-len(source)] if len(source) else f.dom
        cobase = f.cod[:-len(target)] if len(target) else f.cod
        return cls.equation_factory(
            f.then(cobase @ g).trace(len(source)),
            (base @ g).then(f).trace(len(target)))


class ResiduatedMonoid[C0, C1: ResiduatedMonoid](ColouredMonoid[C0, C1]):
    """
    A monoid is residuated when it comes with methods ``over`` and ``under``
    with syntactic sugar ``<<`` and ``>>``.
    """
    @abstractmethod
    def over(self, other: C1) -> C1:
        """ The right-to-left exponential object ``self`` to the ``other``. """

    @abstractmethod
    def under(self, other: C1) -> C1:
        """ The left-to-right exponential object ``self`` to the ``other``. """

    def __lshift__(self, other):
        return self.over(other)

    def __rshift__(self, other):
        return other.under(self)


class BiclosedCategory[C0: ResiduatedMonoid, C1: BiclosedCategory](
        MonoidalCategory[C0, C1], TwoCategory[NoneType, C0, C1]):
    """
    A biclosed category is a :class:`MonoidalCategory` with methods :code:`ev`
    and :code:`curry` for the evaluation and currying of morphisms.

    We also assume the type for objects comes with methods for left and right
    exponentials :code`x << y` and :code`x >> y`, of uncoloured types: a
    biclosed category has the one trivial colour, :class:`NoneType`.
    """
    @classmethod
    @generator
    @abstractmethod
    def ev[Y: Atom[C0], E: Atom[C0], L: Bool](
            cls, base: Y, exponent: E, left: L = True
    ) -> C1[L[(Y << E) @ E, E @ (E >> Y)], Y]:
        """
        The evaluation of an exponential type, to be instantiated: as a
        rule, ``(y << e) @ e ⊢ y`` is a left evaluation and ``e @ (e >>
        y) ⊢ y`` a right one.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """

    @abstractmethod
    def curry(self, n: int = 1, left: bool = True) -> C1:
        """
        The currying of a morphism, to be instantiated.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """

    def base_and_exponent(self, n: int, left: bool) -> tuple[C0, C0]:
        """
        The base and exponent that :meth:`uncurry` evaluates, read off the
        exponential object in the codomain.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if not self.cod.is_exp:
            raise ValueError
        base, exponent = self.cod.base, self.cod.exponent
        if n < len(exponent):
            raise ValueError
        return base, exponent

    def uncurry(self, n: int = 1, left: bool = True) -> C1:
        """
        Uncurry a morphism by composing it with :meth:`ev`, assuming its
        codomain is an exponential object. If the exponent has less than
        ``n`` objects, we uncurry the remaining ones in turn.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if n < 0:
            raise ValueError
        if not n:
            return self
        base, exponent = self.base_and_exponent(n, left)
        result = self @ exponent >> self.ev(base, exponent, True) if left\
            else exponent @ self >> self.ev(base, exponent, False)
        return result.uncurry(n - len(exponent), left)

    @classmethod
    def uncurry_composition(
            cls, f: C1, base: C0, exponent: C0, left: bool) -> C1:
        """
        Curry ``f`` then evaluate it back, i.e. whisker the currying with
        ``exponent`` and compose with :meth:`ev`, the roundtrip that
        :meth:`currying_left` and :meth:`currying_right` state equal to
        ``f``.

        Parameters:
            f : The morphism to curry and evaluate back.
            base : The base of the exponential, i.e. the codomain of ``f``.
            exponent : The objects curried out of the domain of ``f``.
            left : Whether to curry on the left or right.
        """
        curried = f.curry(left=left)
        ev = cls.ev(base, exponent, left)
        return (curried @ exponent).then(ev) if left\
            else (exponent @ curried).then(ev)

    @axiom
    def currying_left[A: C0, X: Atom[C0], E: Atom[C0]](
            cls, f: C1[A @ E, X], base: X, exponent: E) -> Equation[C1]:
        """ Left currying followed by evaluation. """
        return cls.equation_factory(
            cls.uncurry_composition(f, base, exponent, left=True), f)

    @axiom
    def currying_right[A: C0, X: Atom[C0], E: Atom[C0]](
            cls, f: C1[E @ A, X], base: X, exponent: E) -> Equation[C1]:
        """ Right currying followed by evaluation. """
        return cls.equation_factory(
            cls.uncurry_composition(f, base, exponent, left=False), f)


class Pregroup[C0, C1: Pregroup](ResiduatedMonoid[C0, C1]):
    """
    A pregroup is a residuated monoid where the left and right exponentials are
    given by tensoring with the chosen left and right duals for each object.
    """
    l: C1
    r: C1

    def over(self, other: C1) -> C1:
        return self @ other.l

    def under(self, other: C1) -> C1:
        return other.r @ self

    @axiom
    def adjunction(
            cls, x: C1) -> Equation[C1]:
        """ The left and right adjoints are mutually inverse. """
        return cls.equation_factory(x.l.r, x, x.r.l)


class RigidCategory[C0: Pregroup, C1: RigidCategory](BiclosedCategory[C0, C1]):
    """
    A rigid category is a :class:`BiclosedCategory` with a :class:`Pregroup` as
    object type and methods for :code:`cups` and :code:`caps`.
    """
    @classmethod
    @generator
    @abstractmethod
    def cups[X: Atom[C0]](cls, left: X, right: X.r) -> C1[X @ X.r, ()]:
        """
        The cups witnessing :code:`right` as the adjoint of :code:`left`:
        as a rule, ``x @ x.r ⊢ ()`` is a cup, ``x.l @ x`` included.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """

    @classmethod
    @generator
    @abstractmethod
    def caps[X: Atom[C0]](cls, left: X, right: X.l) -> C1[(), X @ X.l]:
        """
        The caps witnessing :code:`right` as the adjoint of :code:`left`:
        as a rule, ``() ⊢ x @ x.l`` is a cap, ``x.r @ x`` included.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """

    @classmethod
    @inapplicable("A rigid evaluation is a cup, which cups reaches.")
    def ev(cls, base: C0, exponent: C0, left: bool = True) -> C1:
        """
        The evaluation of a rigid morphism is obtained using cups.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """
        return base @ cls.cups(exponent.l, exponent) if left\
            else cls.cups(exponent, exponent.r) @ base

    def curry(self, n: int = 1, left: bool = True) -> C1:
        """
        The curry of a rigid morphism is obtained using caps.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """
        if n < 0 or n > len(self.dom):
            raise ValueError
        if not n:
            return self
        if left:
            base, exponent = self.dom[:-n], self.dom[-n:]
            return base @ self.caps(exponent, exponent.l) >> self @ exponent.l
        base, exponent = self.dom[n:], self.dom[:n]
        return self.caps(exponent.r, exponent) @ base >> exponent.r @ self

    def base_and_exponent(self, n: int, left: bool) -> tuple[C0, C0]:
        """
        Contrary to :meth:`BiclosedCategory.base_and_exponent`, a pregroup has
        no exponential object to read the exponent off the codomain: it is the
        ``n`` objects at the end resp. the start of the codomain, dualised.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if n > len(self.cod):
            raise ValueError
        return (self.cod[:-n], self.cod[-n:].r) if left\
            else (self.cod[n:], self.cod[:n].l)

    def transpose(self, left: bool = False) -> C1:
        """
        The transpose of a morphism, i.e. its composition with cups and caps.

        Parameters:
            left : Whether to transpose left or right.

        Example
        -------
        >>> from discopy.monoidal import Equation
        >>> from discopy.rigid import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
        >>> Equation(f.transpose(left=True), f, f.transpose(),
        ...     symbols=("$\\\\mapsfrom$", "$\\\\mapsto$")).draw(
        ...         figsize=(8, 3), doctest="docs/_static/rigid/transpose.svg")

        .. image:: /_static/rigid/transpose.svg
        """
        if left:
            return self.cod.l @ self.caps(self.dom, self.dom.l)\
                >> self.cod.l @ self @ self.dom.l\
                >> self.cups(self.cod.l, self.cod) @ self.dom.l
        return self.caps(self.dom.r, self.dom) @ self.cod.r\
            >> self.dom.r @ self @ self.cod.r\
            >> self.dom.r @ self.cups(self.cod, self.cod.r)

    @axiom
    def snake_equations(
            cls, x: C0) -> Equation[C1]:
        """ The two snake equations. """
        snake_r = (cls.id(x) @ cls.caps(x.r, x)).then(
            cls.cups(x, x.r) @ cls.id(x))
        snake_l = (cls.caps(x, x.l) @ cls.id(x)).then(
            cls.id(x) @ cls.cups(x.l, x))
        return cls.equation_factory(snake_r, cls.id(x), snake_l)

    @axiom
    def caps_coherence[N: NonEmpty[C0], K: NonEmpty[C0]](
            cls, x: N, y: K) -> Equation[C1]:
        """ Monoidal coherence of caps. """
        return cls.equation_factory(
            cls.caps(x @ y, (x @ y).l),
            cls.caps(x, x.l).then(x @ cls.caps(y, y.l) @ x.l))

    @axiom
    def rotate_contravariance[A: C0, B: C0, C: C0](
            cls, f: C1[A, B], g: C1[B, C]) -> Equation[C1]:
        """ Rotation reverses composition. """
        return cls.equation_factory(
            f.then(g).rotate(), g.rotate().then(f.rotate()))


class PivotalCategory[C0, C1](RigidCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A pivotal category is a :class:`RigidCategory` where the left and right
    adjoints coincide, hence it is also a :class:`TracedCategory`.
    """

    @axiom
    def self_dual(
            cls, x: C0) -> Equation[C0]:
        """ Equality of left and right adjoints. """
        return cls.ob.equation_factory(x.r, x.l)

    @axiom
    def pivotality(
            cls, f: C1) -> Equation[C1]:
        """ Equality of left and right transposes. """
        dom, cod = f.dom, f.cod
        left_transpose = (cod.l @ cls.caps(dom, dom.l)).then(
            cod.l @ f @ dom.l).then(cls.cups(cod.l, cod) @ dom.l)
        right_transpose = (cls.caps(dom.r, dom) @ cod.r).then(
            dom.r @ f @ cod.r).then(dom.r @ cls.cups(cod, cod.r))
        return cls.equation_factory(left_transpose, right_transpose)


class BraidedCategory[C0, C1](
        MonoidalCategory[C0, C1], TwoCategory[NoneType, C0, C1]):
    """
    A braided category is a :class:`MonoidalCategory` with a method
    :code:`braid` for the natural isomorphism :code:`x @ y -> y @ x`: its
    wires cross, so it has the one trivial colour, :class:`NoneType`.
    """
    @classmethod
    @generator
    @abstractmethod
    def braid[X: Atom[C0], Y: Atom[C0]](
            cls, left: X, right: Y) -> C1[X @ Y, Y @ X]:
        """
        The braid of two objects, to be instantiated: as a rule, ``x @ y
        ⊢ y @ x`` is a braid over.

        Parameters:
            left : The object on the left of the braid.
            right : The object on the right of the braid.
        """

    @axiom
    def hexagon_left[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: X, y: Y, z: Z) -> Equation[C1]:
        """ The left hexagon equation. """
        return cls.equation_factory(
            cls.braid(x, y @ z),
            (cls.braid(x, y) @ z).then(y @ cls.braid(x, z)))

    @axiom
    def hexagon_right[X: Atom[C0], Y: Atom[C0], Z: Atom[C0]](
            cls, x: X, y: Y, z: Z) -> Equation[C1]:
        """ The right hexagon equation. """
        return cls.equation_factory(
            cls.braid(x @ y, z),
            (x @ cls.braid(y, z)).then(cls.braid(x, z) @ y))

    @axiom
    def braid_naturality(
            cls, f: C1, g: C1) -> Equation[C1]:
        """ Naturality of the braid. """
        return cls.equation_factory(
            f @ g >> cls.braid(f.cod, g.cod),
            cls.braid(f.dom, g.dom) >> g @ f)


class PROB[C1: PROB](PRO[C1], BraidedCategory[Nat, C1]):
    """
    A PROB is a :class:`BraidedCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free braided category on one generator.
    """


class SymmetricCategory[C0, C1](BraidedCategory[C0, C1]):
    """
    A symmetric category is a :class:`BraidedCategory` where the braid is its
    own inverse called :code:`swap` for the symmetry :code:`x @ y -> y @ x`.
    """
    @classmethod
    @abstractmethod
    def swap(cls, left: C0, right: C0) -> C1:
        """
        The swap of two objects, to be instantiated.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[C0]) -> C1:
        """ Compose swaps to permute the atomic objects in ``dom``. """
        xs, doms = list(xs), list(doms)
        if list(range(len(doms))) != sorted(xs):
            raise ValueError
        tensor = lambda objects: cls.ob().tensor(*objects)
        result, done = cls.id(tensor(doms)), cls.ob()
        while xs != list(range(len(xs))):
            i = xs[0]
            left, head = tensor(doms[:i]), tensor(doms[i:i + 1])
            result >>= done @ cls.swap(left, head) @ tensor(doms[i + 1:])
            done, doms = done @ head, doms[:i] + doms[i + 1:]
            xs = [x - 1 if x > i else x for x in xs[1:]]
        return result

    @classmethod
    def braid(cls, left: C0, right: C0) -> C1:
        return cls.swap(left, right)

    @classmethod
    def shuffles(cls, dom, cod) -> list:
        """ The non-identity permutations of ``dom`` with codomain ``cod``. """
        return [
            perm for perm in permutations(range(len(dom)))
            if perm != tuple(range(len(dom)))
            and dom[:0].tensor(*(dom[i:i + 1] for i in perm)) == cod]

    @rule(applies=lambda cls, dom, cod, size:
          size == 1 and len(dom) == len(cod) >= 2 and dom != cod
          and sorted(map(repr, dom)) == sorted(map(repr, cod)))
    def permuting(cls, draw, dom, cod, size, types):
        """ ``x ⊢ y`` is a permutation when ``y`` shuffles ``x``. """
        from hypothesis import strategies as st

        perm = draw(st.sampled_from(cls.shuffles(dom, cod)))
        return [], lambda: cls.permutation_factory(dom, list(perm))

    @permuting.hint
    def permuting(cells, dom, cod):
        """ Either boundary with two adjacent atoms swapped. """
        from hypothesis import strategies as st

        swapped = [
            boundary[:i] @ boundary[i + 1:i + 2] @ boundary[i:i + 1]
            @ boundary[i + 2:]
            for boundary in (dom, cod) for i in range(len(boundary) - 1)
            if boundary[i:i + 1] != boundary[i + 1:i + 2]]
        return st.sampled_from(swapped) if swapped else st.nothing()

    @axiom
    def swap_inverse(
            cls, x: C0, y: C0) -> Equation[C1]:
        """ Involutivity of the swap. """
        return cls.equation_factory(
            cls.swap(x, y).then(cls.swap(y, x)), cls.id(x @ y))


class PROP[C1: PROP](PROB[C1], SymmetricCategory[Nat, C1]):
    """
    A PROP is a :class:`SymmetricCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free symmetric category on one generator.
    """


class MarkovCategory[C0, C1](SymmetricCategory[C0, C1]):
    """
    A Markov category is a :class:`SymmetricCategory` with methods
    :code:`copy` and :code:`merge` for the supply of commutative comonoids.
    """
    @classmethod
    @generator
    @abstractmethod
    def copy[X: Atom[C0], N: Count](cls, x: X, n: N = 2) -> C1[X, X ** N]:
        """
        Make :code:`n` copies of a given object :code:`x`: as a rule,
        ``x ⊢ x @ .. @ x`` is a copy, none or up to three drawn.

        Parameters:
            x : The object to copy.
            n : The number of copies.
        """

    @axiom
    def copy_counitality(
            cls, x: C0) -> Equation[C1]:
        """ Counitality of copying. """
        copy, discard = cls.copy(x), cls.copy(x, n=0)
        return cls.equation_factory(
            copy.then(discard @ x), cls.id(x),
            copy.then(x @ discard))

    @axiom
    def copy_coassociativity(
            cls, x: C0) -> Equation[C1]:
        """ Coassociativity of copying. """
        copy = cls.copy(x)
        return cls.equation_factory(
            copy.then(copy @ x), copy.then(x @ copy))

    @axiom
    def copy_cocommutativity(
            cls, x: C0) -> Equation[C1]:
        """ Cocommutativity of copying. """
        copy = cls.copy(x)
        return cls.equation_factory(copy.then(cls.swap(x, x)), copy)

    @axiom
    def discard_coherence(
            cls, x: C0) -> Equation[C1]:
        """ Monoidal coherence of discarding. """
        return cls.equation_factory(
            cls.copy(x @ x, n=0),
            cls.copy(x, n=0) @ cls.copy(x, n=0))

    @axiom
    def copy_monoidal_coherence(
            cls, x: C0) -> Equation[C1]:
        """ Monoidal coherence of copying. """
        return cls.equation_factory(
            cls.copy(x @ x),
            (cls.copy(x) @ cls.copy(x)).then(
                x @ cls.swap(x, x) @ x))


class ClosedCategory[C0, C1](BiclosedCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A closed category is a symmetric :class:`BiclosedCategory`. We also assume
    it comes with copy and discard so it is also a :class:`MarkovCategory`.
    """


class FeedbackCategory[C0, C1](MarkovCategory[C0, C1]):
    """
    A feedback category is a :class:`MarkovCategory` with a :code:`delay`
    endofunctor and a :code:`feedback` operator.
    """
    @abstractmethod
    def delay(self, n_steps: int = 1) -> C1:
        """
        The delay endofunctor applied to a morphism.

        Parameters:
            n_steps : The number of time steps to delay.
        """

    @rule
    @abstractmethod
    def feedback[A: C0, B: C0, M: Atom[C0]](
            self: C1[A @ M.d, B @ M], dom: A, cod: B, mem: M) -> C1[A, B]:
        """
        The feedback operator on a morphism: as a rule, ``x ⊢ y`` is the
        feedback of ``x @ m.delay() ⊢ y @ m`` over an atom.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to trace over.
        """

    @axiom
    def feedback_vanishing(
            cls, f: C1) -> Equation[C1]:
        """ Vanishing of feedback over the unit. """
        return cls.equation_factory(f.feedback(mem=cls.ob()), f)

    @axiom
    def feedback_joining[A: C0, P: Pair[C0]](
            cls, f: C1[A @ P.d, A @ P], mem: P) -> Equation[C1]:
        """ Joining nested feedback loops. """
        return cls.equation_factory(
            f.feedback(mem=mem), f.feedback().feedback())

    dagger_involution = Category.dagger_involution.inapplicable(
        "The delay of a feedback category is not reversible.")

    dagger_contravariance = Category.dagger_contravariance.inapplicable(
        "The delay of a feedback category is not reversible.")


class BalancedCategory[C0, C1](
        BraidedCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A balanced category is a :class:`BraidedCategory` and a
    :class:`TracedCategory` with a method :code:`twist` for the natural
    automorphism :code:`x -> x`.
    """
    @classmethod
    @generator
    @abstractmethod
    def twist[X: Atom[C0]](cls, dom: X) -> C1[X, X]:
        """
        The twist on an object, to be instantiated. As a rule, ``x ⊢ x``
        is a twist.

        Parameters:
            dom : The object on which to take the twist.
        """

    @axiom
    def balanced_twist[X: Atom[C0], Y: Atom[C0]](
            cls, x: X, y: Y) -> Equation[C1]:
        """ Compatibility of the twist and braid. """
        return cls.equation_factory(
            cls.twist(x @ y),
            cls.braid(x, y).then(
                cls.twist(y) @ cls.twist(x)).then(
                    cls.braid(y, x)))


class RibbonCategory[C0, C1](
        PivotalCategory[C0, C1], BalancedCategory[C0, C1]):
    """
    A ribbon category is a :class:`PivotalCategory` which is also a
    :class:`BalancedCategory`, i.e. where diagrams can draw knots and links.
    """

    @axiom
    def twist_as_trace[X: Atom[C0]](
            cls, x: X) -> Equation[C1]:
        """ The twist as both orientations of a traced braid. """
        braid = cls.braid(x, x)
        return cls.equation_factory(
            braid.trace(left=True), cls.twist(x), braid.trace())


class CompactCategory[C0, C1](
        RibbonCategory[C0, C1], SymmetricCategory[C0, C1]):
    """
    A compact category is a :class:`RibbonCategory` which is also a
    :class:`SymmetricCategory`, i.e. with cups, caps and swaps and where
    the twist is the identity.
    """
    @classmethod
    @inapplicable("The twist is the identity.")
    def twist(cls, dom: C0) -> C1:
        return cls.id(dom)

    @axiom
    def reidemeister_1_cap(
            cls, x: C0) -> Equation[C1]:
        """ Reidemeister move 1 for caps. """
        return cls.equation_factory(
            cls.caps(x, x.r).then(cls.swap(x, x.r)),
            cls.caps(x.r, x))

    @axiom
    def reidemeister_1_cup(
            cls, x: C0) -> Equation[C1]:
        """ Reidemeister move 1 for cups. """
        return cls.equation_factory(
            cls.swap(x, x.r).then(cls.cups(x.r, x)),
            cls.cups(x, x.r))


class HypergraphCategory[C0, C1](
        CompactCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A hypergraph category is a symmetric category with a supply of spiders,
    i.e. special commutative Frobenius algebras on each objects.

    This makes it both a :class:`CompactCategory` and a :class:`MarkovCategory`
    """
    @classmethod
    @generator
    @abstractmethod
    def spiders[X: Atom[C0], M: Count, N: Count](
            cls, n_legs_in: M, n_legs_out: N, typ: X) -> C1[X ** M, X ** N]:
        """
        The spiders on a given type with ``n_legs_in`` and ``n_legs_out``:
        as a rule, ``x @ .. @ x ⊢ x @ .. @ x`` is a spider, on up to three
        legs a side drawn.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
