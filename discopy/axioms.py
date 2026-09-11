"""
Property-based testing of the axioms with `Hypothesis
<https://hypothesis.readthedocs.io>`_.

An :class:`Axiom` is an equation stated once on a :class:`Testable`
class — an abstract base class of :mod:`discopy.abc`, whether a category
or the serialisation interface — and inherited by every class below it,
where
:meth:`Axiom.failing` and :meth:`Axiom.inapplicable` classify it when a
class breaks it or has no such structure. A class that implements
:meth:`Testable.strategy` generates its own terms; one that cannot do so
yet leaves the strategy to raise. The matrix in ``proptest/`` reads the
types it quantifies over off :meth:`Testable.subclasses` rather than a
list, and checks every axiom of every type against generated arguments,
one cell per pair; CONTRIBUTING.md says how to run it.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Equation
    Axiom
    AxiomFailure
    Testable
    Grid
    ComposablePair
    ComposableTriple
    Serialisable

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
import pickle
import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, ClassVar, Concatenate, Self, TypeVar

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
:meth:`Serialisable.repr_transparency`, in its :attr:`Axiom.scope` when it
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
    :meth:`Serialisable.repr_transparency`.

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


class Testable[T]:
    """
    A testable class states axioms, which its subclasses inherit along
    with the structure they axiomatise, and says how to generate the
    terms those axioms quantify over.

    Both kinds of law meet here: a :class:`discopy.abc.Category` states
    those of a categorical structure, a
    :class:`Serialisable` those of writing a term down and
    reading it back. A class need not be a category to state laws, which
    is why the two meet here rather than in either of them, nor need it
    be either: :class:`ComposablePair` states the law it enforces on the
    pairs it generates.

    A type that implements :meth:`strategy` is one the property matrix
    in ``proptest/`` quantifies over, checking each of its axioms against
    generated terms. One that does not is not checked, and says so by
    leaving :meth:`strategy` to raise.
    """

    @classmethod
    def strategy(cls, **params) -> st.SearchStrategy[T]:
        """
        Build a `search strategy
        <https://hypothesis.readthedocs.io/en/latest/data.html>`_ for
        instances of ``cls``, which is how a class enrols itself in the
        property matrix.

        An override that delegates to another strategy accepts
        ``**params``, pops the parameters it consumes and forwards the
        rest, so that a caller's bounds pass through unchanged and a
        subclass overrides what a base popped just by passing it. A
        terminal strategy instead declares exactly the parameters it
        implements: a constraint it cannot honour fails loudly as an
        unexpected keyword rather than being silently dropped.

        The default raises: a class states its laws as soon as it has
        them, and is checked against them once it says how to draw their
        terms. It is deliberately not an :func:`abc.abstractmethod`,
        which would make every category that has not implemented one
        uninstantiable rather than merely unchecked.

        >>> from discopy.monoidal import Diagram
        >>> Diagram.strategy()
        Traceback (most recent call last):
         ...
        NotImplementedError: No search strategy implemented for Diagram
        """
        raise NotImplementedError(
            f"No search strategy implemented for {cls.__name__}")

    @classproperty
    def axioms(cls) -> dict[str, Axiom]:
        """
        The axioms inherited by ``cls``, keyed by name and bound to
        ``cls``, an override on a subclass hiding the base it overrides.

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
    def subclasses(cls) -> tuple[type[Testable], ...]:
        """
        Every transitive subclass of ``cls``, ``cls`` itself included.

        A subclass is listed once, however many paths reach it, in the
        order a breadth-first walk of the subclass graph first meets
        it.

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


no_strategy = Testable.__dict__["strategy"]
"""
The default :meth:`Testable.strategy`, under a name that can be assigned.

A class inherits it from :class:`Testable` unless a base it refines
implements one: the terms of a :class:`discopy.monoidal.Ty` are not
those of the :class:`discopy.cat.Ob` it subclasses, so a class that
would inherit the wrong strategy declares ``strategy = no_strategy``
until it implements its own.
"""


class Grid(Testable, NamedGeneric["factory"], tuple):
    """ A rectangular grid with composable rows and columns. """

    n_rows: ClassVar[int]
    n_columns: ClassVar[int]
    n_active_rows: ClassVar[int] = 1

    @axiom
    def composability(cls, grid: Self) -> Equation:
        """
        Every cell composes with the cell below it, the law that
        :meth:`__new__` enforces and :meth:`strategy` draws for: the
        codomains of each row are the domains of the row under it.
        """
        above = tuple(
            grid[i].cod for i in range(cls.n_columns * (cls.n_rows - 1)))
        below = tuple(grid[i + cls.n_columns].dom for i in range(len(above)))
        return Equation(above, below)

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
        """
        Generate a grid column-by-column using composable boundaries.

        A grid draws its cells from its ``factory``, so an unsubscripted
        one has no strategy: ``ComposablePair`` cannot generate anything,
        ``ComposablePair[Arrow]`` generates pairs of arrows.
        """
        from hypothesis import strategies as st

        factory = cls.factory
        if factory is None:
            raise NotImplementedError(
                f"No search strategy implemented for {cls.__name__}")
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


class Serialisable(Testable):
    """
    The serialisation interface of DisCoPy, one hook driving all three
    mechanisms: the class attribute ``serialised_attrs`` names attributes that
    are also keyword arguments of ``__init__``, from which follow

    - a generic pair of inverse methods :meth:`to_tree` and
      :meth:`from_tree`, the JSON serialisation behind
      :func:`discopy.utils.dumps` and :func:`discopy.utils.loads`,
    - a generic :meth:`__repr__` such that ``eval(repr(x)) == x``,
    - :meth:`__setstate__`, the terminal of every pickle migration
      chain; the class parameters of
      :class:`discopy.utils.NamedGeneric` are pickled by their own
      machinery.

    A subclass with a different constructor declares its keys once
    instead of reimplementing each method.

    Each mechanism comes with the law that it is a roundtrip, i.e. that
    a term reads back from what it was written to:
    :meth:`repr_transparency`
    for its representation, :meth:`pickling` and :meth:`copying` for the
    pickle protocol and :meth:`serialisation` for its tree. They are
    axioms like any other, so a class that also implements
    :meth:`discopy.axioms.Testable.strategy` has them checked against
    generated terms, and one that violates a law declares it
    ``.failing`` rather than leaving it untested.

    Example
    -------
    >>> from discopy.cat import Box
    >>> assert Box.serialised_attrs\\
    ...     == ('name', 'dom', 'cod', 'is_dagger', 'data')
    >>> f = Box('f', 'x', 'y', data=42)
    >>> assert Box.from_tree(f.to_tree()) == f
    """
    serialised_attrs: tuple[str, ...] = ()

    def is_default(self, key: str) -> bool:
        """
        Whether the value of an attribute equals its class default,
        in which case :meth:`to_tree` and :meth:`__repr__` drop it.

        Parameters:
            key : The name of the attribute.
        """
        if not hasattr(type(self), key):
            return False
        value, default = getattr(self, key), getattr(type(self), key)
        return value is default or (
            type(value) is type(default) and value == default)

    def __repr__(self):
        """
        The transparent representation of a DisCoPy object: an attribute
        without a class default is positional, one that differs from its
        default is a keyword argument and one equal to it is dropped.

        Example
        -------
        >>> import discopy
        >>> from discopy.cat import Box
        >>> f = Box('f', 'x', 'y', data=42)
        >>> f
        cat.Box('f', cat.Ob('x'), cat.Ob('y'), data=42)
        >>> assert eval(repr(f), vars(discopy)) == f
        """
        return factory_name(type(self)) + "(" + ", ".join(
            f"{key}={repr(getattr(self, key))}" if hasattr(type(self), key)
            else repr(getattr(self, key))
            for key in self.serialised_attrs if not self.is_default(key)) + ")"

    def __setstate__(self, state):
        """
        Restore a pickled state, the terminal that every pickle
        migration shim chains into with ``super().__setstate__``.

        Parameters:
            state : The pickled state of the object.
        """
        self.__dict__.update(state)

    def to_tree(self) -> dict:
        """
        Serialise a DisCoPy object, see :func:`dumps`.

        The tree records the :func:`factory_name` and then each of the
        ``serialised_attrs``, dropping a key when its value equals the
        class attribute of the same name, e.g. a box that is not a
        dagger. An attribute with a ``to_tree`` method is serialised, a
        non-empty list or tuple of such attributes becomes the list of
        their trees, raw JSON data passes through unchanged.

        Example
        -------
        >>> from pprint import PrettyPrinter
        >>> pprint = PrettyPrinter(indent=4, width=70, sort_dicts=False).pprint
        >>> from discopy.cat import Box
        >>> f = Box('f', 'x', 'y', data=42)
        >>> pprint((f >> f[::-1]).to_tree())
        {   'factory': 'cat.Arrow',
            'inside': [   {   'factory': 'cat.Box',
                              'name': 'f',
                              'dom': {'factory': 'cat.Ob', 'name': 'x'},
                              'cod': {'factory': 'cat.Ob', 'name': 'y'},
                              'data': 42},
                          {   'factory': 'cat.Box',
                              'name': 'f',
                              'dom': {'factory': 'cat.Ob', 'name': 'y'},
                              'cod': {'factory': 'cat.Ob', 'name': 'x'},
                              'is_dagger': True,
                              'data': 42}],
            'dom': {'factory': 'cat.Ob', 'name': 'x'},
            'cod': {'factory': 'cat.Ob', 'name': 'x'}}
        """
        tree = {'factory': factory_name(type(self))}
        for key in self.serialised_attrs:
            if self.is_default(key):
                continue
            value = getattr(self, key)
            if hasattr(value, 'to_tree'):
                value = value.to_tree()
            elif isinstance(value, (list, tuple)) and value and all(
                    hasattr(v, 'to_tree') for v in value):
                value = [v.to_tree() for v in value]
            tree[key] = value
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Serialisable:
        """
        Decode a serialised DisCoPy object, see :func:`loads`.

        A key missing from the tree falls back to the default value of
        the corresponding keyword argument of ``__init__``. A value with
        a ``'factory'`` key decodes recursively, a non-empty list of
        such values to the tuple of decoded objects, raw JSON data
        passes through unchanged.

        Parameters:
            tree : DisCoPy serialisation.

        Example
        -------
        >>> from discopy.cat import Ob
        >>> assert Ob.from_tree({'factory': 'cat.Ob', 'name': 'x'}) == Ob('x')
        """
        from discopy.utils import from_tree

        kwargs = {}
        for key in cls.serialised_attrs:
            if key not in tree:
                continue
            value = tree[key]
            if isinstance(value, dict) and 'factory' in value:
                value = from_tree(value)
            elif isinstance(value, list) and value and all(
                    isinstance(v, dict) and 'factory' in v for v in value):
                value = tuple(map(from_tree, value))
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def environment(cls) -> dict:
        """
        The namespace the representation of a term reads back in: the
        public names of the package, as ``from discopy import *`` binds
        them, so that a representation qualified by module such as
        ``cat.Box('f', cat.Ob('x'), cat.Ob('y'))`` evaluates, and then
        those of the module the class is defined in, so that one
        printing bare names such as ``Tensor[int]([0], dom=Dim(1),
        cod=Dim(1))`` evaluates too. The module comes second because a
        term prints the names its own module binds: ``Dim`` in
        ``discopy.tensor`` is the one a tensor is built from.

        The import is local because the package imports this module.
        """
        import discopy

        public = lambda namespace: {
            name: value for name, value in namespace.items()
            if not name.startswith("_")}
        module = sys.modules[cls.__module__]
        return dict(public(vars(discopy)), **public(vars(module)))

    @axiom
    def repr_transparency(cls, term: Self) -> Equation:
        """
        The representation of a term evaluates back to it, in the
        :meth:`environment` of its type.

        Its ``str`` reads back too, which STYLE.md asks of every term
        under "the obvious variable naming convention", but the
        environment that convention needs is more than a namespace, so
        ``str_transparency`` is left to
        `#764 <https://github.com/discopy/discopy/issues/764>`_.
        """
        return Equation(eval(repr(term), cls.environment()), term)

    @axiom
    def pickling(cls, term: Self) -> Equation:
        """
        A term loads back from its pickle, of the same class: the equation
        is between the pairs of a class and a term, since a subscript of a
        :class:`discopy.utils.NamedGeneric` is part of what a pickle keeps.
        """
        loaded = pickle.loads(pickle.dumps(term))
        return Equation((type(loaded), loaded), (type(term), term))

    @axiom
    def copying(cls, term: Self) -> Equation:
        """
        A term is equal to its deep copy, of the same class. Copying goes
        through the same protocol as :meth:`pickling` without the bytes,
        so a class whose reduction drops what its state needs — the
        parameters of a :class:`discopy.utils.NamedGeneric`, say — breaks
        one law with the other.
        """
        copied = deepcopy(term)
        return Equation((type(copied), copied), (type(term), term))

    @axiom
    def serialisation(cls, term: Self) -> Equation:
        """
        A term decodes back from its tree and from the JSON of its tree.
        A type without a tree declares the law inapplicable.
        """
        from discopy.utils import dumps, from_tree, loads

        return Equation(from_tree(term.to_tree()), loads(dumps(term)), term)
