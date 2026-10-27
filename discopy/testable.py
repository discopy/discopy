"""
Property-based testing of the axioms with `Hypothesis
<https://hypothesis.readthedocs.io>`_.

An :class:`Axiom` is an equation stated once on an abstract base class of
:mod:`discopy.abc` and inherited by every category below it, where
:meth:`Axiom.failing` and :meth:`Axiom.inapplicable` classify it when a
category breaks it or has no such structure. A :class:`Testable` category
generates its own objects and arrows, and states the laws every type that
does so obeys: a term reads back from its representation, its pickle and
its tree. The matrix in ``proptest/`` checks every axiom of every
category against generated arguments, one cell per pair; CONTRIBUTING.md
says how to run it.

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
    Atomic
    NonEmpty
    BoundaryConnected
    HorizontalPair
    Square
    TraceSuperposing
    TraceSliding
    TraceNaturalityLeft
    TraceNaturalityRight
    TraceDinaturality
    TraceDinaturalityLeft
    TraceDinaturalityRight
    Currying
    LeftCurrying
    RightCurrying
    FeedbackVanishing
    FeedbackJoining

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
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, ClassVar, Concatenate, Self, TypeVar

from discopy.utils import (
    AxiomError,
    NamedGeneric,
    assert_iscomposable,
    dumps,
    factory_name,
    from_tree,
    get_origin,
    loads,
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
        def law(_category):
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
    its instances, and the laws every such type obeys: a term reads back
    from its representation, its pickle and its tree.
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

    @classmethod
    def environment(cls) -> dict:
        """
        The namespace the representation of a term reads back in: the
        public names of the package, as ``from discopy import *`` binds
        them, so that a representation qualified by module such as
        ``cat.Box('f', cat.Ob('x'), cat.Ob('y'))`` evaluates, and then
        those of the module the category is defined in, so that one
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
    def transparency(cls, term: Self) -> Equation:
        """
        The representation of a term evaluates back to it, in the
        :meth:`environment` of its type.
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
    def serialisation(cls, term: Self) -> Equation:
        """
        A term decodes back from its tree and from the JSON of its tree.
        A type without a tree declares the law inapplicable.
        """
        return Equation(from_tree(term.to_tree()), loads(dumps(term)), term)


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


@dataclass(frozen=True)
class Atomic(Testable, NamedGeneric["factory"]):
    """ An object of the factory with exactly one generator inside. """

    value: C0

    def __post_init__(self):
        if len(self.value) != 1:
            raise ValueError("Expected an atomic object.")

    @classmethod
    def strategy(cls, **params):
        """ Generate an object with exactly one generator inside. """
        return resolve(cls.factory, **params).filter(
            lambda value: len(value) == 1).map(cls)


@dataclass(frozen=True)
class NonEmpty(Testable, NamedGeneric["factory"]):
    """ A non-empty object of the factory. """

    value: C0

    def __post_init__(self):
        if not self.value:
            raise ValueError("Expected a non-empty object.")

    @classmethod
    def strategy(cls, **params):
        """ Generate a non-empty object. """
        return resolve(cls.factory, **params).filter(bool).map(cls)


@dataclass(frozen=True)
class BoundaryConnected(Testable, NamedGeneric["factory"]):
    """
    An arrow whose boundary reaches every box, read through its map when it
    is not one itself, or a grid of such arrows, connected cell by cell.
    """

    value: C1

    def __post_init__(self):
        cells = self.value if isinstance(self.value, Grid)\
            else (self.value, )
        for cell in cells:
            graph = cell if hasattr(cell, "is_boundary_connected")\
                else cell.to_map()
            if not graph.is_boundary_connected:
                raise ValueError("Expected a boundary-connected arrow.")

    @classmethod
    def strategy(cls, **params):
        """ Generate from the factory restricted to connected arrows. """
        return resolve(
            cls.factory, boundary_connected=True, **params).map(cls)


class HorizontalPair(Grid):
    """ Two arrows side by side, the arguments of a tensor. """

    n_rows, n_columns = 1, 2


class Square(Grid):
    """ A two-by-two grid of arrows, the arguments of the interchange law. """

    n_rows = n_columns = 2


class TraceSuperposing(Testable, NamedGeneric["factory"], tuple):
    """ A traceable arrow and an object to superpose on it. """

    def __new__(cls, traced: C1, obj: C0):
        traced.trace()
        return super().__new__(cls, (traced, obj))

    @classmethod
    def strategy(cls):
        """ Generate the identity on an atomic object, and an object. """
        from hypothesis import strategies as st

        atoms = resolve(cls.factory.ob).filter(lambda obj: len(obj) == 1)
        return st.tuples(atoms, resolve(cls.factory.ob)).map(
            lambda pair: cls(cls.factory.id(pair[0]), pair[1]))


class TraceSliding(Testable, NamedGeneric["factory"], tuple):
    """
    An arrow traced over an object, on the side :attr:`left` says, and
    one sliding around the trace on that object.
    """

    left: ClassVar[bool]

    def __new__(cls, traced: C1, obj: C0, sliding: C1):
        traced_dom = obj @ sliding.cod if cls.left else sliding.cod @ obj
        traced_cod = obj @ sliding.dom if cls.left else sliding.dom @ obj
        if (traced.dom, traced.cod) != (traced_dom, traced_cod):
            raise ValueError("Expected compatible trace sliding boundaries.")
        return super().__new__(cls, (traced, obj, sliding))

    @classmethod
    def strategy(cls):
        """ Generate two arrows with compatible trace boundaries. """
        from hypothesis import strategies as st

        factory, objects = cls.factory, resolve(cls.factory.ob)
        traced = resolve(factory.ob, min_length=1)

        def arrows(args):
            obj, dom, cod = args
            traced_dom = obj @ cod if cls.left else cod @ obj
            traced_cod = obj @ dom if cls.left else dom @ obj
            return st.tuples(
                factory.strategy(
                    dom=traced_dom, cod=traced_cod, min_leaves=1),
                factory.strategy(dom=dom, cod=cod, min_leaves=1)).map(
                    lambda pair: cls(pair[0], obj, pair[1]))

        return st.tuples(traced, objects, objects).flatmap(arrows)


class TraceNaturalityLeft(TraceSliding):
    """ The arguments of left trace naturality. """

    left = True


class TraceNaturalityRight(TraceSliding):
    """ The arguments of right trace naturality. """

    left = False


class TraceDinaturality(Testable, NamedGeneric["factory"], tuple):
    """
    An arrow and one sliding around its trace, traceable only once the
    sliding arrow is composed in on either side.
    """

    left: ClassVar[bool]

    def __new__(cls, traced: C1, sliding: C1):
        traced_in, traced_out = (
            (traced.dom[:len(sliding.cod)], traced.cod[:len(sliding.dom)])
            if cls.left else
            (traced.dom[len(traced.dom) - len(sliding.cod):],
             traced.cod[len(traced.cod) - len(sliding.dom):]))
        if (traced_in, traced_out) != (sliding.cod, sliding.dom):
            raise ValueError("Expected compatible trace sliding boundaries.")
        return super().__new__(cls, (traced, sliding))

    @classmethod
    def strategy(cls):
        """ Generate an arrow sliding between two traced objects. """
        from hypothesis import strategies as st

        factory, objects = cls.factory, resolve(cls.factory.ob)
        traced = resolve(factory.ob, min_length=1)

        def arrows(args):
            base, cobase, source, target = args
            traced_dom = source @ base if cls.left else base @ source
            traced_cod = target @ cobase if cls.left else cobase @ target
            return st.tuples(
                factory.strategy(
                    dom=traced_dom, cod=traced_cod, min_leaves=1),
                factory.strategy(dom=target, cod=source, min_leaves=1)).map(
                    lambda pair: cls(*pair))

        return st.tuples(objects, objects, traced, traced).flatmap(arrows)


class TraceDinaturalityLeft(TraceDinaturality):
    """ The arguments of left trace dinaturality. """

    left = True


class TraceDinaturalityRight(TraceDinaturality):
    """ The arguments of right trace dinaturality. """

    left = False


class Currying(Testable, NamedGeneric["factory"], tuple):
    """
    An evaluation with its base and exponent, to curry on the side
    :attr:`left` says.
    """

    left: ClassVar[bool]

    def __new__(cls, arrow: C1, base: C0, exponent: C0):
        exponential = base << exponent if cls.left else exponent >> base
        arrow_dom = exponential @ exponent if cls.left\
            else exponent @ exponential
        if (arrow.dom, arrow.cod) != (arrow_dom, base):
            raise ValueError("Expected an evaluation.")
        return super().__new__(cls, (arrow, base, exponent))

    @classmethod
    def strategy(cls):
        """ Generate the evaluation of two atomic objects. """
        from hypothesis import strategies as st

        atoms = resolve(cls.factory.ob).filter(lambda obj: len(obj) == 1)
        return st.tuples(atoms, atoms).map(lambda pair: cls(
            cls.factory.ev(*pair, left=cls.left), *pair))


class LeftCurrying(Currying):
    """ An evaluation with its base and exponent, to curry on the left. """

    left = True


class RightCurrying(Currying):
    """ An evaluation with its base and exponent, to curry on the right. """

    left = False


class FeedbackVanishing(Testable, NamedGeneric["factory"], tuple):
    """ An arrow and the unit, the memory of a feedback that vanishes. """

    def __new__(cls, arrow: C1, unit: C0):
        if len(unit):
            raise ValueError("Expected the monoidal unit.")
        arrow.feedback(mem=unit)
        return super().__new__(cls, (arrow, unit))

    @classmethod
    def strategy(cls, **params):
        """ Generate an arrow with the unit beside it. """
        return cls.factory.strategy(**params).map(
            lambda arrow: cls(arrow, cls.factory.ob()))


class FeedbackJoining(Testable, NamedGeneric["factory"], tuple):
    """ An arrow with at least two units of memory to feed back. """

    def __new__(cls, arrow: C1, memory: C0):
        if len(memory) < 2:
            raise ValueError("Expected at least two units of memory.")
        if arrow.dom[-len(memory):] != memory.delay():
            raise ValueError("Expected the delayed memory in the domain.")
        if arrow.cod[-len(memory):] != memory:
            raise ValueError("Expected the memory in the codomain.")
        return super().__new__(cls, (arrow, memory))

    @classmethod
    def strategy(cls):
        """ Generate an arrow over two atomic units of memory. """
        from hypothesis import strategies as st

        objects = resolve(cls.factory.ob)
        atoms = objects.filter(lambda obj: len(obj) == 1)

        def arrows(args):
            obj, first, second = args
            memory = first @ second
            return cls.factory.strategy(
                dom=obj @ memory.delay(), cod=obj @ memory).map(
                    lambda arrow: cls(arrow, memory))

        return st.tuples(objects, atoms, atoms).flatmap(arrows)


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


def assert_generators(diagram: type) -> None:
    """
    Check that one of each generator of a free category, as its own
    :meth:`strategy` draws it, satisfies its :meth:`serialisation` and
    :meth:`transparency` laws, with the same single shot as
    :func:`assert_axioms`.

    Parameters:
        diagram : The diagram class whose :attr:`generators` are checked.
    """
    from hypothesis import Phase, find, settings

    single_shot = settings(
        max_examples=1, phases=(Phase.generate, ), database=None)
    for generator in diagram.generators:
        box = find(
            generator.strategy(), lambda _: True, settings=single_shot)
        for law in (generator.serialisation, generator.transparency):
            verdict = law(box)
            assert verdict is NotImplemented or verdict, law
