"""Data structures and strategies for property tests."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from typing import ClassVar, TypeVar, TYPE_CHECKING, get_args, get_origin

from discopy.utils import AxiomError, assert_iscomposable

if TYPE_CHECKING:
    from hypothesis import strategies as st


C0 = TypeVar("C0")
C1 = TypeVar("C1")
"""
The object and arrow types of the carrier an axiom is bound to.

An axiom annotates its arguments with these rather than with the concrete
types of the module it is written in, so that a subclass inherits the
override with its own types: :func:`proptest.strategies.arguments` rebinds
both names to ``carrier.ob`` and ``carrier.ar`` when it evaluates the
annotations. This is also why every module stating an axiom needs
``from __future__ import annotations``, which keeps them unevaluated.
"""


GENERATORS = tuple("abcde")
"""
The names the generators of a free category are drawn from.

They are finitely many and shared, so a generated functor can name every one
of them: composing two functors keeps only the keys of the left-hand map, so
a functor that named just a few would compose to one defined nowhere else.
"""


class Strategy[T](ABC):
    """
    A type with a canonical property-test strategy.
    Using ``hypothesis``, we can get the default search strategy dispatch
    through any object that defines a method called ``draw``, but this
    would conflict with our existing ``draw`` methods, so we do it manually
    with this custom trait.
    """

    @classmethod
    @abstractmethod
    def strategy(cls, **params) -> "st.SearchStrategy[T]":  # pragma: no cover
        """Build a strategy for instances of ``cls``."""


class Natural(int, Strategy["Natural"]):
    """ A non-negative integer with tensor given by addition. """

    def __new__(cls, value=0):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Expected a non-negative integer.")
        return super().__new__(cls, value)

    def __matmul__(self, other):
        return type(self)(self + other) if isinstance(other, int)\
            else NotImplemented

    __rmatmul__ = __matmul__
    __len__ = lambda self: int(self)

    @classmethod
    def equation_factory(cls, *terms):
        """ Construct an equation between natural numbers. """
        from discopy.cat import Equation

        return Equation(*terms)

    @classmethod
    def strategy(cls, *, max_size=3):
        """Generate non-negative integers."""
        from hypothesis import strategies as st

        return st.one_of(
            st.just(1),
            st.integers(min_value=0, max_value=max_size)).map(cls)


@dataclass(frozen=True)
class Atomic[T](Strategy[T]):
    """ An object containing exactly one generator. """

    value: T

    def __post_init__(self):
        if len(self.value) != 1:
            raise ValueError("Expected an atomic object.")

    @classmethod
    def strategy(cls, *, factory: type[T]):
        """Generate an object containing exactly one generator."""
        return factory.strategy().filter(
            lambda value: len(value) == 1).map(cls)


@dataclass(frozen=True)
class NonEmpty[T](Strategy[T]):
    """ A non-empty object. """

    value: T

    def __post_init__(self):
        if not len(self.value):
            raise ValueError("Expected a non-empty object.")

    @classmethod
    def strategy(cls, *, factory: type[T], **params):
        """Generate a non-empty object."""
        return factory.strategy(**params).filter(bool).map(cls)


class PastingDiagram[T](Strategy[tuple[T, ...]], tuple[T, ...]):
    """ A rectangular grid with composable rows and columns. """

    n_rows: ClassVar[int]
    n_columns: ClassVar[int]
    n_active_rows: ClassVar[int] = 1

    def __new__(cls, *cells: T):
        if len(cells) != cls.n_rows * cls.n_columns:
            raise ValueError("Expected one value per cell.")
        for row in range(cls.n_rows - 1):
            for column in range(cls.n_columns):
                i = row * cls.n_columns + column
                assert_iscomposable(cells[i], cells[i + cls.n_columns])
        for row in range(cls.n_rows):
            for column in range(cls.n_columns - 1):
                i = row * cls.n_columns + column
                cells[i] @ cells[i + 1]
        return super().__new__(cls, cells)

    @classmethod
    def strategy(cls, *, factory: type[T], **params):
        """Generate a grid column-by-column using composable boundaries."""
        from hypothesis import strategies as st

        dom, cod = params.pop("dom", None), params.pop("cod", None)

        @st.composite
        def pasting_diagram(draw):
            active = draw(st.integers(
                min_value=0,
                max_value=cls.n_rows - cls.n_active_rows))
            columns = []
            for _ in range(cls.n_columns):
                column, boundary = [], dom
                for row in range(cls.n_active_rows):
                    cell = draw(factory.strategy(
                        dom=boundary,
                        cod=cod if row == cls.n_active_rows - 1 else None,
                        **params))
                    column.append(cell)
                    boundary = cell.cod
                columns.append(
                    active * [factory.id(column[0].dom)]
                    + column
                    + (cls.n_rows - active - cls.n_active_rows)
                    * [factory.id(column[-1].cod)])
            return cls(*(
                columns[column][row]
                for row in range(cls.n_rows)
                for column in range(cls.n_columns)))

        return pasting_diagram()


class ComposablePair[T](PastingDiagram[T], tuple[T, T]):
    """ Two morphisms composable from left to right. """

    n_rows, n_columns = 2, 1
    n_active_rows = 2


class ComposableTriple[T](PastingDiagram[T], tuple[T, T, T]):
    """ Three values composable from left to right. """

    n_rows, n_columns = 3, 1
    n_active_rows = 3


class HorizontalPair[T](PastingDiagram[T], tuple[T, T]):
    """ Two horizontally composable cells. """

    n_rows, n_columns = 1, 2


class Bifunctor[T](PastingDiagram[T], tuple[T, T, T, T]):
    """ A two-by-two pasting diagram for bifunctoriality. """

    n_rows = n_columns = 2


class TraceSuperposing[C0, C1](
        Strategy[tuple[C1, C0]], tuple[C1, C0]):
    """ A traceable arrow and an object to superpose. """

    def __new__(cls, traced: C1, obj: C0):
        traced.trace()
        return super().__new__(cls, (traced, obj))

    @classmethod
    def strategy(cls, *, factory: type[C1]):
        """Generate a traceable identity and an arbitrary object."""
        from hypothesis import strategies as st

        object_type, arrow_type = factory.ob, factory
        objects = object_type.strategy()
        atomic = object_type.strategy().filter(lambda obj: len(obj) == 1)
        return st.tuples(atomic, objects).map(
            lambda pair: cls(arrow_type.id(pair[0]), pair[1]))


class TraceSliding[C0, C1](
        Strategy[tuple[C1, C0, C1]], tuple[C1, C0, C1]):
    """ Arguments for trace sliding over an arbitrary traced type. """

    left: ClassVar[bool]

    def __new__(cls, traced: C1, obj: C0, sliding: C1):
        traced_dom = obj @ sliding.cod if cls.left else sliding.cod @ obj
        traced_cod = obj @ sliding.dom if cls.left else sliding.dom @ obj
        if (traced.dom, traced.cod) != (traced_dom, traced_cod):
            raise ValueError("Expected compatible trace sliding boundaries.")
        return super().__new__(cls, (traced, obj, sliding))

    @classmethod
    def strategy(cls, *, factory: type[C1]):
        """Generate non-trivial morphisms with compatible trace boundaries."""
        from hypothesis import strategies as st

        objects = factory.ob.strategy()
        traced = factory.ob.strategy(min_length=1)

        def morphisms(args):
            obj, dom, cod = args
            traced_dom = obj @ cod if cls.left else cod @ obj
            traced_cod = obj @ dom if cls.left else dom @ obj
            return st.tuples(
                factory.strategy(
                    dom=traced_dom, cod=traced_cod, min_leaves=1),
                factory.strategy(dom=dom, cod=cod, min_leaves=1)).map(
                    lambda pair: cls(pair[0], obj, pair[1]))

        return st.tuples(traced, objects, objects).flatmap(morphisms)


class TraceNaturalityLeft[C0, C1](TraceSliding[C0, C1]):
    """ Arguments for left-oriented trace naturality. """

    left = True


class TraceNaturalityRight[C0, C1](TraceSliding[C0, C1]):
    """ Arguments for right-oriented trace naturality. """

    left = False


class TraceDinaturality[C0, C1](
        Strategy[tuple[C1, C1]], tuple[C1, C1]):
    """ A traceable arrow and an arrow to slide around its trace. """

    left: ClassVar[bool]

    def __new__(cls, traced: C1, sliding: C1):
        traced_in, traced_out = (
            (traced.dom[:len(sliding.cod)], traced.cod[:len(sliding.dom)])
            if cls.left else
            (traced.dom[-len(sliding.cod):], traced.cod[-len(sliding.dom):]))
        if (traced_in, traced_out) != (sliding.cod, sliding.dom):
            raise ValueError("Expected compatible trace sliding boundaries.")
        return super().__new__(cls, (traced, sliding))

    @classmethod
    def strategy(cls, *, factory: type[C1]):
        """Generate an arrow sliding between two traced objects."""
        from hypothesis import strategies as st

        objects = factory.ob.strategy()
        traced = factory.ob.strategy(min_length=1)

        def arrows(args):
            base, cobase, source, target = args
            traced_dom = source @ base if cls.left else base @ source
            traced_cod = target @ cobase if cls.left else cobase @ target
            return st.tuples(
                factory.strategy(
                    dom=traced_dom, cod=traced_cod, min_leaves=1),
                factory.strategy(
                    dom=target, cod=source, min_leaves=1)).map(
                        lambda pair: cls(*pair))

        return st.tuples(objects, objects, traced, traced).flatmap(arrows)


class TraceDinaturalityLeft[C0, C1](TraceDinaturality[C0, C1]):
    """ Arguments for left-oriented trace dinaturality. """

    left = True


class TraceDinaturalityRight[C0, C1](TraceDinaturality[C0, C1]):
    """ Arguments for right-oriented trace dinaturality. """

    left = False


class LeftCurrying[C0, C1](
        Strategy[tuple[C1, C0, C0]], tuple[C1, C0, C0]):
    """ Arguments for left currying followed by evaluation. """

    left = True

    def __new__(cls, arrow: C1, base: C0, exponent: C0):
        exponential = base << exponent if cls.left else exponent >> base
        arrow_dom = exponential @ exponent if cls.left\
            else exponent @ exponential
        if (arrow.dom, arrow.cod) != (arrow_dom, base):
            raise ValueError("Expected an evaluation morphism.")
        return super().__new__(cls, (arrow, base, exponent))

    @classmethod
    def strategy(cls, *, factory: type[C1]):
        """Generate an evaluation suitable for left or right currying."""
        from hypothesis import strategies as st

        object_type, arrow_type = factory.ob, factory
        objects = object_type.strategy().filter(lambda obj: len(obj) == 1)
        return st.tuples(objects, objects).map(lambda pair: cls(
            arrow_type.ev(*pair, left=cls.left), *pair))


class RightCurrying[C0, C1](LeftCurrying[C0, C1]):
    """ Arguments for right currying followed by evaluation. """

    left = False


class FeedbackVanishing[C0, C1](
        Strategy[tuple[C1, C0]], tuple[C1, C0]):
    """ A feedback arrow together with the monoidal unit. """

    def __new__(cls, arrow: C1, unit: C0):
        if len(unit):
            raise ValueError("Expected the monoidal unit.")
        arrow.feedback(mem=unit)
        return super().__new__(cls, (arrow, unit))

    @classmethod
    def strategy(cls, *, factory: type[C1], **params):
        """Generate a feedback arrow paired with the monoidal unit."""
        object_type, arrow_type = factory.ob, factory
        return arrow_type.strategy(**params).map(
            lambda arrow: cls(arrow, object_type()))


class FeedbackJoining[C0, C1](
        Strategy[tuple[C1, C0]], tuple[C1, C0]):
    """ A feedback arrow with at least two units of memory. """

    def __new__(cls, arrow: C1, memory: C0):
        if len(memory) < 2:
            raise ValueError("Expected at least two units of memory.")
        if arrow.dom[-len(memory):] != memory.delay():
            raise ValueError("Expected the delayed memory in the domain.")
        if arrow.cod[-len(memory):] != memory:
            raise ValueError("Expected the memory in the codomain.")
        return super().__new__(cls, (arrow, memory))

    @classmethod
    def strategy(cls, *, factory: type[C1]):
        """Generate a feedback arrow with two units of memory."""
        from hypothesis import strategies as st

        object_type, arrow_type = factory.ob, factory
        objects = object_type.strategy()
        atomic = object_type.strategy().filter(lambda obj: len(obj) == 1)

        def arrows(args):
            obj, first, second = args
            memory = first @ second
            return arrow_type.strategy(
                dom=obj @ memory.delay(), cod=obj @ memory).map(
                    lambda arrow: cls(arrow, memory))

        return st.tuples(objects, atomic, atomic).flatmap(arrows)


@dataclass(frozen=True, eq=False)
class Relabelling(Mapping):
    """
    A map on the generators of a free category, sending the atoms it names to
    a chosen object and every other one to itself.

    It is a :class:`Mapping` rather than a closure so that functors built
    from it can be composed and compared, which is what makes the axioms of
    ``Cat`` itself checkable: :meth:`discopy.utils.MappingOrCallable.then`
    composes by iterating the keys of the left-hand map, and equality
    compares the wrapped maps. Iterating yields only the atoms it renames,
    while looking one up is total, so a functor built from it applies to any
    diagram and still composes to something comparable.
    """
    images: tuple[tuple[object, object], ...] = ()

    def __getitem__(self, atom):
        """
        The image of an atomic object, carrying over whatever the atom does:
        a rotation in a rigid category, a delay in a feedback one.
        """
        wire, = getattr(atom, "inside", (atom, ))
        for key, image in self.images:
            other, = getattr(key, "inside", (key, ))
            if other.name == wire.name:
                break
        else:
            return atom
        turns = getattr(wire, "z", 0)
        for _ in range(abs(turns)):
            image = image.l if turns < 0 else image.r
        steps = getattr(wire, "time_step", 0)
        return image.delay(steps) if steps else image

    def __iter__(self):
        return iter([key for key, _ in self.images])

    def __len__(self):
        return len(self.images)

    def __bool__(self):
        """ A relabelling is total, even when it renames nothing. """
        return True

    def send(self, typ):
        """ The image of an object, atom by atom. """
        if not hasattr(typ, "inside"):
            return self[typ]
        return type(typ)().tensor(*(
            self[typ[i:i + 1]] for i in range(len(typ))))


@dataclass(frozen=True, eq=False)
class Relabelled(Mapping):
    """
    Send each box to one of the same name on the relabelled boundary.

    Boxes cannot be enumerated, so this iterates empty and two of them
    compare equal as mappings do. That is what lets a functor built from a
    :class:`Relabelling` be the unit of its own composition on the right.
    """
    objects: Relabelling

    def __getitem__(self, box):
        return type(box)(
            box.name, self.objects.send(box.dom), self.objects.send(box.cod))

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __bool__(self):
        """ A relabelling is total, even though it enumerates nothing. """
        return True


class Axiom[T]:
    """
    A categorical law, stated either of a carrier or of one of its elements.

    An axiom whose first parameter is ``cls`` is a law of the category: it is
    bound to the carrier and its remaining arguments are generated. One whose
    first parameter is ``self`` is a law of an element, e.g. a functor, so the
    element is generated too and the law reads as a method on it.

    Calling a bound axiom returns its own verdict: :obj:`NotImplemented` when
    the structure does not apply to the carrier, an
    :class:`discopy.utils.AxiomError` wrapping the equation when the law is
    known to be broken, and the equation itself otherwise.

    A law is broken when *some* argument is a counterexample, not every one,
    so :attr:`broken` is the flag :meth:`failing` stamps on the body rather
    than a verdict read for particular arguments — the property matrix reads
    it before generating anything and searches for the counterexample.
    """

    def __init__(self, equation, *, carrier=None, name=None):
        function = equation.__func__ if isinstance(equation, classmethod)\
            else equation
        self.equation = function
        self.signature = inspect.signature(function)
        self.receiver = next(iter(self.signature.parameters), None)
        self.carrier = carrier
        self.name = self.__name__ = name or function.__name__
        self.broken = getattr(function, "broken", False)
        self.__doc__ = function.__doc__

    def __repr__(self):
        return f"Axiom({self.name})"

    def __set_name__(self, owner, name):
        """
        Take the name of the attribute the axiom is assigned to, so that an
        override built with :meth:`modulo`, :meth:`failing` or
        :meth:`inapplicable` needs no name of its own.
        """
        self.name = self.__name__ = name

    @property
    def is_method(self) -> bool:
        """ Whether the law is stated of an element rather than a carrier. """
        return self.receiver == "self"

    def bind(self, carrier: type[T]) -> Axiom[T]:
        """ Bind the axiom to a concrete carrier. """
        return type(self)(self.equation, carrier=carrier, name=self.name)

    def __get__(self, instance, owner: type[T]) -> Axiom[T]:
        return self.bind(owner)

    def modulo(self, up_to) -> Axiom[T]:
        """
        The same law with its equation compared up to a function, so that a
        carrier weakens an inherited axiom in one statement, e.g.
        ``bifunctoriality = MonoidalCategory.bifunctoriality.modulo(
        normal_form)``.
        """
        @wraps(self.equation)
        def equation(*args, **kwargs):
            return self.equation(*args, **kwargs).modulo(up_to)
        return type(self)(equation)

    def failing(self, reason: str) -> Axiom[T]:
        """
        The same law declared broken, its equation wrapped in an
        :class:`discopy.utils.AxiomError` with the reason as message and
        documentation, e.g. ``braid_naturality =
        BraidedCategory.braid_naturality.failing("A free braid is a box.")``.
        """
        @wraps(self.equation)
        def equation(*args, **kwargs):
            return AxiomError(reason, self.equation(*args, **kwargs))
        equation.__doc__, equation.broken = reason, True
        return type(self)(equation)

    def inapplicable(self, reason: str) -> Axiom[T]:
        """
        The same law declared not to apply to the carrier: it takes no
        argument and returns :obj:`NotImplemented`, with the reason as its
        documentation, e.g. ``trace_vanishing =
        TracedCategory.trace_vanishing.inapplicable("No trace.")``.
        """
        def law(cls):
            return NotImplemented
        law.__name__, law.__doc__ = self.name, reason
        return type(self)(law)

    def strategy(self) -> "st.SearchStrategy":
        """
        Generate the arguments the bound axiom expects.

        ``C0`` and ``C1`` resolve to the objects and arrows of the carrier,
        or of the carrier's domain for a law of an element: the arguments a
        functor is applied to live in the category it maps from, and its
        codomain is reachable as ``self.cod`` from the body.
        """
        from hypothesis import strategies as st

        function = inspect.unwrap(self.equation)
        source = self.carrier.dom if self.is_method else self.carrier
        scope = {"C0": source.ob, "C1": source.ar}
        annotations = inspect.get_annotations(
            function, globals=function.__globals__, locals=scope,
            eval_str=True)
        annotations[self.receiver] = self.carrier
        required = (
            parameter for parameter in self.parameters
            if parameter.default is inspect.Parameter.empty)
        return st.tuples(*(
            resolve(annotations[parameter.name]) for parameter in required))

    def falsify(self, **params) -> tuple:
        """
        Search for a shrunk counterexample to the bound axiom: arguments for
        which the verdict fails — the equation is false, or the
        implementation refuses to build its terms — raising
        :class:`hypothesis.errors.NoSuchExample` when no counterexample is
        found. Keyword arguments are passed to :func:`hypothesis.find`.

        >>> from discopy.matrix import Matrix
        >>> Matrix[int].copy_cocommutativity.falsify()
        (2,)
        >>> Matrix[int].unitality.falsify()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
         ...
        hypothesis.errors.NoSuchExample: No examples found of condition ...
        """
        from hypothesis import find

        if self.carrier is None:
            raise TypeError(f"{self.name} is not bound to a class.")

        def refutes(args):
            try:
                verdict = self(*args)
            except Exception:
                return True
            return verdict is not NotImplemented and not holds(verdict)

        return find(self.strategy(), refutes, **params)

    @property
    def parameters(self) -> tuple[inspect.Parameter, ...]:
        """
        The parameters whose arguments the property matrix generates.

        For a law of an element that includes the element itself, so an axiom
        that takes none states its verdict before anything is generated.
        """
        explicit = tuple(self.signature.parameters.values())[1:]
        if not self.is_method:
            return explicit
        receiver = inspect.Parameter(
            self.receiver, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=self.carrier)
        return (receiver, ) + explicit

    def __call__(self, *args, **kwargs):
        if self.carrier is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        signature = self.signature.replace(parameters=self.parameters)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if self.is_method:
            return self.equation(**bound.arguments)
        return self.equation(
            **{self.receiver: self.carrier, **bound.arguments})


def axiom(equation) -> Axiom:
    """ Decorate an equation as an inherited categorical axiom. """
    return Axiom(equation)


def resolve(annotation, **params) -> "st.SearchStrategy":
    """ Resolve the strategy implemented by an annotated type. """
    origin = get_origin(annotation) or annotation
    if not isinstance(origin, type) or not issubclass(origin, Strategy):
        raise TypeError(
            f"Expected a Strategy annotation, got {annotation!r}.")
    if args := get_args(annotation):
        params["factory"] = args[-1]
    return origin.strategy(**params)


def assert_axioms(*carriers) -> None:
    """
    Check every axiom of each carrier on a single generated example, a dry
    run of the property tests in ``proptest/``.

    An axiom that does not apply is skipped, a broken one is only required
    to state its :class:`discopy.utils.AxiomError` — one example need not
    be a counterexample — and any other law must hold.
    """
    from hypothesis import Phase, find, settings

    single_shot = settings(
        max_examples=1, phases=(Phase.generate, ), database=None)
    for carrier in carriers:
        for axiom in carrier.axioms:
            if not axiom.parameters and axiom() is NotImplemented:
                continue
            args = find(
                axiom.strategy(), lambda value: True, settings=single_shot)
            try:
                verdict = axiom(*args)
            except AxiomError:
                assert axiom.broken, axiom
                continue
            if axiom.broken:
                assert isinstance(verdict, AxiomError), axiom
            else:
                assert holds(verdict), axiom


def assert_strategy_finds(carrier, *structures) -> None:
    """
    Check that the strategy of an arrow carrier generates a term containing
    a box of each of the given structural classes.
    """
    from hypothesis import find

    for structure in structures:
        find(carrier.strategy(), lambda term: any(
            isinstance(box, structure)
            for box in getattr(term, "boxes", term.inside)))


def assert_verdict(axiom: Axiom, verdict) -> None:
    """
    Assert the verdict a bound axiom returned for some arguments.

    An :class:`discopy.utils.AxiomError` wraps the equation of a law that is
    known to be broken — as its last argument, after an optional reason —
    and carries none at all when the implementation refused to build its
    terms. Either way the equation is asserted: it is :attr:`Axiom.broken`
    that tells the runner to expect the failure.
    """
    assert holds(verdict)


def holds(verdict) -> bool:
    """
    Whether a verdict asserts, unwrapping the equation a broken law carries
    as the last argument of its :class:`discopy.utils.AxiomError`.
    """
    if isinstance(verdict, AxiomError):
        verdict = verdict.args[-1] if verdict.args else False
    return bool(verdict)


def declared_axioms(cls) -> dict[str, Axiom]:
    """
    The axioms a class declares, by name, subclasses overriding bases.

    Names are collected before they are filtered, so that assigning anything
    that is not an axiom over an inherited one drops it altogether, rather
    than restating it.
    """
    visible = {
        name: value
        for base in reversed(cls.__mro__)
        for name, value in base.__dict__.items()}
    return {name: value for name, value in visible.items()
            if isinstance(value, Axiom)}
