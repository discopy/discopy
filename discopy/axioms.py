"""
Property-based testing of the axioms with `Hypothesis
<https://hypothesis.readthedocs.io>`_.

An :class:`Axiom` is an equation stated once on a :class:`Theory` — an
abstract base class of :mod:`discopy.abc`, whether a category or the
serialisation interface — and inherited by every class below it, where
:meth:`Axiom.failing` and :meth:`Axiom.inapplicable` classify it when a
class breaks it or has no such structure. A :class:`Testable` class
generates its own instances; one that does not yet declares
:data:`no_strategy` as its axioms, which is how it opts out. The matrix
in ``proptest/`` reads its carriers off :meth:`Theory.subclasses` rather
than a list, and checks every axiom of every carrier against generated
arguments, one cell per pair; CONTRIBUTING.md says how to run it.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Equation
    Axiom
    AxiomFailure
    Theory
    Testable
    Grid
    ComposablePair
    ComposableTriple

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        axiom
        resolve
        substitute
        assert_axioms
"""

from __future__ import annotations

import __future__
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, ClassVar, Concatenate, TypeVar

from discopy.utils import (
    AxiomError,
    NamedGeneric,
    assert_iscomposable,
    classproperty,
    factory_name,
    get_origin,
)

if TYPE_CHECKING:
    from hypothesis import strategies as st


C0 = TypeVar("C0")
C1 = TypeVar("C1")
"""
The object and arrow types of the category an axiom is bound to.

An axiom annotates its arguments with these rather than with the concrete
types of the module it is written in, so that a subclass inherits the
override with its own types: :meth:`Axiom.strategy` rebinds both names to
``category.ob`` and ``category.ar``, and :data:`typing.Self` to the category
itself for a law of every term of a type whatever its level, such as
:meth:`Testable.transparency`, in its :attr:`Axiom.scope` when it
evaluates the annotations; a law of functors names the category they map
from as ``Self.dom``. This is also why every module stating an axiom
needs ``from __future__ import annotations``, which keeps them
unevaluated: :class:`Axiom` refuses an equation compiled without it.
Rebinding happens through the ``locals`` of that evaluation because the
:pep:`695` type parameters of :class:`discopy.abc.Category` live in a
scope :func:`eval` cannot see, in globals or anywhere else.
"""


GENERATORS = tuple("abcde")
"""
The names the generators of a free category are drawn from.

They are finitely many and shared, so a generated functor can name every one
of them: composing two functors keeps only the keys of the left-hand map, so
a functor that named just a few would compose to one defined nowhere else.
"""


class Equation(NamedGeneric["ar"]):
    """
    An equation is a list of ``terms`` to be compared up to a function
    ``up_to``, the identity by default.  Casting it to ``bool`` checks
    whether its terms are all equal up to that function.

    Parameters:
        terms : The terms of the equation.
        symbol : The symbol between each pair of terms, ``"="`` by default.
        symbols : The symbols between each pair of terms, overriding
            ``symbol``; ``len(terms) * (symbol, )`` by default.
        up_to : The function up to which ``bool(equation)`` compares its
            terms, overriding the subclass' :attr:`up_to` if given.

    Example
    -------
    The number of boxes inside an arrow is left unchanged by associativity,
    so we can compare arrows up to the function that counts them modulo 2:

    >>> from discopy.cat import Ob, Box, Equation
    >>> x = Ob('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> parity = lambda term: len(term.inside) % 2
    >>> assert not Equation(f, f >> g >> g)
    >>> assert Equation(f, f >> g >> g, up_to=parity)
    """
    up_to = None

    def __init__(self, *terms, symbol="=", symbols=None, up_to=None):
        self.terms = terms
        self.symbols = tuple(symbols) if symbols is not None\
            else len(terms) * (symbol, )
        if up_to is not None:
            self.up_to = up_to

    def modulo(self, up_to: Callable) -> Equation:
        """
        The same equation compared up to the given function, rebinding
        :attr:`up_to`, whose name the attribute already takes.

        >>> from discopy.cat import Ob, Box, Equation
        >>> x = Ob('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> assert Equation(f >> g, g >> f).modulo(lambda _: True)
        """
        return type(self)(*self.terms, symbols=self.symbols, up_to=up_to)

    def __repr__(self):
        """
        >>> from discopy.cat import Ob, Box, Equation
        >>> Equation(Box('f', Ob('x'), Ob('x')))
        cat.Equation(cat.Box('f', cat.Ob('x'), cat.Ob('x')))
        """
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.terms))})"

    def __str__(self):
        return f"Equation({', '.join(map(str, self.terms))})"

    def __bool__(self):
        terms = self.terms if self.up_to is None\
            else list(map(self.up_to, self.terms))
        return all(term == terms[0] for term in terms)


class AxiomFailure(AxiomError):
    """
    A law declared broken, raised when the bound axiom is called: the
    reason is the message and :attr:`equation` is the law evaluated on the
    arguments, whose sides say how it failed.
    """

    def __init__(self, reason: str, equation):
        super().__init__(reason, equation)
        self.equation = equation


@dataclass
class Axiom[**P, T]:
    """
    An axiom of a category, stated once on an abstract base class and
    inherited by every category below it.

    The category is the class the axiom is bound to, e.g.
    :class:`discopy.cat.Arrow` for the axioms of
    :class:`discopy.abc.Category`. The axiom is a classmethod of it,
    implicitly: its first parameter is the category and the remaining ones
    are generated from their annotations — an object for the typing of
    identities, three composable arrows for the associativity of
    composition, a term of the category itself for
    :meth:`Testable.transparency`.

    Calling a bound axiom returns its own verdict: :obj:`NotImplemented`
    when the structure does not apply to the category, and the equation
    itself otherwise; a law declared broken raises an
    :class:`AxiomFailure` carrying that equation instead of returning it.

    A law is broken when *some* argument is a counterexample, not every one,
    so :attr:`broken` is declared by :meth:`failing` before any argument is
    generated — the property matrix marks such an axiom as an expected
    failure and lets the search find the counterexample.

    Parameters:
        equation : The function stating the law, from the category and the
            arguments annotated with :obj:`C0`, :obj:`C1`,
            :data:`typing.Self` or a :class:`Testable` to an
            :class:`Equation`, or to :obj:`NotImplemented` when the
            structure does not apply.
        category : The class the axiom is bound to, :obj:`None` until
            :meth:`bind` or the attribute access on a class binds it.
        name : The attribute the law is stored under, the name of the
            equation by default.
        subspaces : The strategies the named parameters are generated from
            instead of their annotations, declared by :meth:`weaken`.
        broken : Whether the law is declared broken by :meth:`failing`.
    """

    equation: Callable[Concatenate[type[T], P], Equation]
    _: KW_ONLY
    category: type[T] = None
    name: str = None
    subspaces: dict = None
    broken: bool = False

    def __post_init__(self):
        function = inspect.unwrap(self.equation)
        deferred = __future__.annotations.compiler_flag
        if not function.__code__.co_flags & deferred:
            raise TypeError(
                f"{function.__module__} states the axiom {function.__name__} "
                "without `from __future__ import annotations`.")
        self.name = self.name or self.equation.__name__
        self.subspaces = dict(self.subspaces or {})
        self.__doc__ = self.equation.__doc__

    def __repr__(self):
        """
        A bound axiom is the attribute of its category, e.g.
        ``cat.Arrow.unitality``; an unbound one wraps a function and has no
        transparent representation.
        """
        if self.category is None:
            return f"{type(self).__name__}({self.name})"
        return f"{factory_name(self.category)}.{self.name}"

    def __hash__(self):
        return hash((self.equation, self.category, self.name))

    def bind(self, category: type[T]) -> Axiom[P, T]:
        """ Bind the axiom to a concrete category. """
        return replace(self, category=category)

    def __get__(self, instance, owner: type[T]) -> Axiom[P, T]:
        return self.bind(owner)

    def modulo(self, up_to) -> Axiom[P, T]:
        """
        The same law with its equation compared up to a function, so that a
        category weakens an inherited axiom in one statement, e.g. a diagram
        compares the interchange law up to its normal form:
        ``Diagram.bifunctoriality = MonoidalCategory.bifunctoriality.modulo(
        Diagram.normal_form)``.
        """
        @wraps(self.equation)
        def equation(*args, **kwargs):
            return self.equation(*args, **kwargs).modulo(up_to)
        return replace(self, equation=equation)

    def failing(self, reason: str) -> Axiom[P, T]:
        """
        The same law declared broken: calling it raises an
        :class:`AxiomFailure` with the reason as message and the equation
        evaluated on the arguments, e.g. ``braid_naturality =
        BraidedCategory.braid_naturality.failing("A free braid is a box.")``.
        """
        @wraps(self.equation)
        def equation(*args, **kwargs):
            raise AxiomFailure(reason, self.equation(*args, **kwargs))
        equation.__doc__ = reason
        return replace(self, equation=equation, broken=True)

    def inapplicable(self, reason: str) -> Axiom[P, T]:
        """
        The same law declared not to apply to the category: it takes no
        argument and returns :obj:`NotImplemented`, with the reason as its
        documentation, e.g. ``trace_vanishing =
        TracedCategory.trace_vanishing.inapplicable("No trace.")``.
        """
        def law(cls):
            return NotImplemented
        law.__doc__ = reason
        return replace(self, equation=law, subspaces={}, broken=False)

    def weaken(self, **subspaces) -> Axiom[P, T]:
        """
        The same law quantified over a subspace of the named arguments,
        e.g. ``unitality_of_loops = Category.unitality.weaken(f=Endo[C1])``
        for a wrapper ``Endo`` of the endomorphisms: each named parameter
        is generated from its subspace strategy, whose wrapper validates
        membership on construction and is unwrapped before the body reads
        it. Assigned to
        its own attribute beside a ``.failing`` declaration, it shows the
        matrix one expected failure and one green cell instead of one
        blanket expected failure.
        """
        return replace(self, subspaces=dict(self.subspaces, **subspaces))

    @property
    def parameters(self) -> tuple[inspect.Parameter, ...]:
        """
        The parameters whose arguments the property matrix generates: all
        but the first, which is the category.
        """
        return tuple(
            inspect.signature(self.equation).parameters.values())[1:]

    @property
    def scope(self) -> dict:
        """
        What the names in the annotations of the axiom stand for: the
        category itself for :data:`typing.Self`, and its objects and arrows
        for :obj:`C0` and :obj:`C1`. A monoid, having no objects of its
        own, stands for both; a class of functors is the arrows of ``Cat``,
        and the category it maps from, where the arguments a functor is
        applied to live, is reachable as ``Self.dom``.
        """
        return {
            "Self": self.category,
            "C0": getattr(self.category, "ob", self.category),
            "C1": getattr(self.category, "ar", self.category)}

    def strategy(self) -> st.SearchStrategy:
        """
        Generate the arguments the bound axiom expects: one per required
        parameter, from its annotation evaluated in the :attr:`scope` of the
        category or from the subspace :meth:`weaken` declared for it.

        Only the parameters' annotations are evaluated: the law's return
        annotation may name a type its module imports for checking only.
        """
        from hypothesis import strategies as st

        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        namespace = inspect.unwrap(self.equation).__globals__
        annotations = {
            parameter.name: eval(parameter.annotation, namespace, self.scope)
            if isinstance(parameter.annotation, str) else parameter.annotation
            for parameter in self.parameters}
        annotations.update({
            name: substitute(annotation, self.scope)
            for name, annotation in self.subspaces.items()})
        return st.tuples(*(
            resolve(annotations[parameter.name])
            for parameter in self.parameters
            if parameter.default is inspect.Parameter.empty))

    def falsify(self, **params) -> tuple:
        """
        Search for a shrunk counterexample to the bound axiom: arguments for
        which the verdict fails — the equation is false, or the
        implementation refuses to build its terms — raising
        :class:`hypothesis.errors.NoSuchExample` when no counterexample is
        found. Keyword arguments are passed to :func:`hypothesis.find`.

        >>> from discopy.cat import Arrow
        >>> Arrow.associativity.falsify()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
         ...
        hypothesis.errors.NoSuchExample: No examples found of condition ...
        """
        from hypothesis import find

        def refutes(args):
            try:
                verdict = self(*args)
            except AxiomFailure:
                return True
            return verdict is not NotImplemented and not verdict

        return find(self.strategy(), refutes, **params)

    def arguments(self, *args: P.args, **kwargs: P.kwargs) -> dict:
        """
        Bind the arguments to the :attr:`parameters` of the bound axiom,
        unwrapping those a :attr:`subspaces` wrapper validated on
        construction.
        """
        if self.category is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        bound = inspect.Signature(self.parameters).bind(*args, **kwargs)
        bound.apply_defaults()
        return {
            name: value.value if name in self.subspaces else value
            for name, value in bound.arguments.items()}

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Equation[T]:
        return self.equation(self.category, **self.arguments(*args, **kwargs))


def axiom[**P, T](
        equation: Callable[Concatenate[type[T], P], Equation]) -> Axiom[P, T]:
    """
    Decorate an equation as a categorical axiom: a classmethod of its
    category, implicitly, whose remaining parameters are generated.
    """
    return Axiom(equation)


class Testable[T](ABC):
    """
    A type that comes with a `search strategy
    <https://hypothesis.readthedocs.io/en/latest/data.html>`_ generating
    its instances, so that the axioms quantifying over them have
    something to quantify over.

    Generating a type is independent from writing it down: the laws that
    a term reads back from its representation, its pickle and its tree
    are those of :class:`discopy.abc.Serialisable`, which a testable
    type inherits when it is serialisable and does not when it is not.
    """

    @classmethod
    @abstractmethod
    def strategy(cls, **params) -> st.SearchStrategy[T]:
        """
        Build a strategy for instances of ``cls``.

        An override that delegates to another strategy accepts
        ``**params``, pops the parameters it consumes and forwards the
        rest, so that a caller's bounds pass through unchanged and a
        subclass overrides what a base popped just by passing it. A
        terminal strategy instead declares exactly the parameters it
        implements: a constraint it cannot honour fails loudly as an
        unexpected keyword rather than being silently dropped.
        """


class Grid(Testable, NamedGeneric["factory"], tuple):
    """ A rectangular grid with composable rows and columns. """

    n_rows: ClassVar[int]
    n_columns: ClassVar[int]
    n_active_rows: ClassVar[int] = 1

    def __new__(cls, *cells: C1):
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
    def strategy(cls, **params):
        """Generate a grid column-by-column using composable boundaries."""
        from hypothesis import strategies as st

        factory = cls.factory
        dom, cod = params.pop("dom", None), params.pop("cod", None)

        @st.composite
        def pasting_diagram(draw):
            """ Draw each column as a chain of cells padded by identities. """
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


class ComposablePair(Grid):
    """ Two morphisms composable from left to right. """

    n_rows, n_columns = 2, 1
    n_active_rows = 2


class ComposableTriple(Grid):
    """ Three values composable from left to right. """

    n_rows, n_columns = 3, 1
    n_active_rows = 3


class Theory:
    """
    A theory is a class that states axioms, which its subclasses inherit
    along with the structure they axiomatise.

    Both kinds of theory subclass this: a :class:`discopy.abc.Category`
    states the laws of a categorical structure, a
    :class:`discopy.abc.Serialisable` those of writing a term down and
    reading it back. A class need not be a category to state laws, which
    is why the two meet here rather than in either of them.

    Stating a law and generating the terms it quantifies over are
    independent axes, so a theory need not be :class:`Testable` and a
    testable type need not be a theory. Every abstract base class of
    :mod:`discopy.abc` states laws with no terms of its own to generate,
    and :class:`discopy.python.Function` — a concrete category whose
    morphisms are Python functions — states them although its equality is
    intensional, so ``f >> id == f`` is false however the function is
    drawn and only :meth:`Axiom.modulo` extensional behaviour can hold.
    Conversely :class:`ComposablePair` generates the arguments a law
    quantifies over while stating no law of its own.
    """

    @classproperty
    def axioms(cls) -> dict[str, Axiom]:
        """
        The axioms inherited by ``cls``, by name, subclasses overriding
        bases and each bound to ``cls``.

        Names are collected before they are filtered, so that assigning
        anything that is not an axiom over an inherited one drops it
        altogether, rather than restating it.
        """
        visible = {
            name: value
            for base in reversed(cls.__mro__)
            for name, value in base.__dict__.items()}
        return {name: value.bind(cls) for name, value in visible.items()
                if isinstance(value, Axiom)}

    @classmethod
    def subclasses(cls) -> tuple[type[Theory], ...]:
        """
        Every transitive subclass of ``cls``, itself included, in the
        order a breadth-first walk of the subclass graph meets them,
        each listed once however many paths reach it.

        Example
        -------
        >>> from discopy.cat import Arrow, Box
        >>> assert Arrow.subclasses()[0] is Arrow
        >>> assert Box in Arrow.subclasses()  # a subclass of a subclass
        """
        found, queue = {cls: None}, [cls]
        while queue:
            for subclass in queue.pop(0).__subclasses__():
                if subclass not in found:
                    found[subclass] = None
                    queue.append(subclass)
        return tuple(found)


declared_axioms = Theory.__dict__["axioms"]
"""
The default :attr:`Theory.axioms`, under a name that a class enrolling
itself below one that opted out can assign back, e.g. ``cat.Ob``.
"""


@classproperty
def no_strategy(cls) -> dict[str, Axiom]:
    """
    The :attr:`Theory.axioms` of a class that does not generate its own
    terms yet, raising :class:`NotImplementedError` rather than listing
    laws that nothing can be drawn to check.

    A class states its laws as soon as it has them and enrols itself in
    the property matrix when it says how to generate the terms they
    quantify over: until then it declares ``axioms = no_strategy``, which
    its subclasses inherit until one of them implements
    :meth:`Testable.strategy` and declares :data:`declared_axioms` back.
    """
    raise NotImplementedError(
        f"No search strategy implemented for {cls.__name__}")


def resolve(annotation, **params) -> st.SearchStrategy:
    """ Resolve the strategy implemented by an annotated type. """
    if not isinstance(annotation, type)\
            or not issubclass(annotation, Testable):
        raise TypeError(
            f"Expected a Testable annotation, got {annotation!r}.")
    return annotation.strategy(**params)


def substitute(annotation, scope: dict):
    """
    Replace the :obj:`C0` and :obj:`C1` type variables of a subspace
    annotation by the objects and arrows they stand for, rebuilding each
    parameterised wrapper whose factory the substitution changes.
    """
    if isinstance(annotation, TypeVar):
        return scope[annotation.__name__]
    factory = getattr(annotation, "factory", None)
    if factory is None or factory is annotation:
        return annotation
    substituted = substitute(factory, scope)
    if substituted is factory:
        return annotation
    return get_origin(annotation)[substituted]


def assert_axioms(*categories) -> None:
    """
    Check every axiom of each category on a single generated example, a dry
    run of the property tests in ``proptest/``.

    An axiom that does not apply is skipped, a broken one is only required
    to raise its :class:`discopy.utils.AxiomError` — one example need not
    be a counterexample — and any other law must hold.
    """
    from hypothesis import Phase, find, settings

    single_shot = settings(
        max_examples=1, phases=(Phase.generate, ), database=None)
    for category in categories:
        for axiom in category.axioms.values():
            if not axiom.parameters and axiom() is NotImplemented:
                continue
            args = find(
                axiom.strategy(), lambda value: True, settings=single_shot)
            try:
                verdict = axiom(*args)
            except AxiomError:
                assert axiom.broken, axiom
            else:
                assert verdict is NotImplemented or verdict, axiom
