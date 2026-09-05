"""
Property-based testing of the axioms with `Hypothesis
<https://hypothesis.readthedocs.io>`_: a law is stated once as an
:class:`Axiom` of an abstract base class, a carrier generates its own
instances through :class:`Strategy`, and the matrix in ``proptest/``
searches every cell for a counterexample.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Axiom
    AxiomFailure
    Strategy
    Natural
    Atomic
    NonEmpty
    Subsingleton
    BoundaryConnected
    PastingDiagram
    ComposablePair
    ComposableTriple
    HorizontalPair
    Square
    TraceSuperposing
    TraceSliding
    TraceNaturalityLeft
    TraceNaturalityRight
    TraceDinaturality
    TraceDinaturalityLeft
    TraceDinaturalityRight
    LeftCurrying
    RightCurrying
    FeedbackVanishing
    FeedbackJoining
    HomogeneousMemory
    Relabelling

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        axiom
        resolve
        substitute
        assert_axioms
        assert_strategy_finds

How to develop DisCoPy against its property suite: state the laws before
writing the implementation, let the matrix search for counterexamples,
replay a failure deterministically, record the counterexample so the bug
can never come back unnoticed, and audit the search strategy whenever a
bug escapes it.

The suite
---------

- ``proptest/test_axioms.py`` is the matrix: every :class:`Axiom` of every
  carrier in ``CARRIERS``, one pytest cell per pair, arguments generated
  by :meth:`Axiom.strategy` from the annotations of the law's own
  parameters.
- :class:`Strategy` states the laws of any type that generates its own
  instances, whatever its level: :meth:`Strategy.transparency`,
  :meth:`Strategy.pickling` and :meth:`Strategy.serialisation` are cells
  of the matrix for every carrier, and a carrier whose representations
  print bare names overrides :meth:`Strategy.environment` with the
  namespace they read back in.
- ``proptest/test_drawing.py`` and ``proptest/test_normal_form.py`` check
  the remaining ad-hoc properties — drawing does not raise, a normal form
  and a foliation are idempotent — over the diagram carriers.
- ``proptest/test_counterexamples.py`` replays every recorded
  counterexample deterministically — no generation, no search: the
  matrix's explicit phase. Its memory is Hypothesis's example database,
  ``.hypothesis`` on your machine and a workflow artifact on CI, which
  every run reads before it searches.
- Select cells by glob: ``uv run pytest proptest/ --axioms '<glob>'
  -vrsxX``, with ``*`` as the only wildcard so brackets match themselves.
  Recorded counterexamples carry the id of their matrix cell, so a glob
  selects a law's search and its records together.
- Each ``test/<module>.py`` gains a ``test_axioms`` dry run (one example
  per axiom, see :func:`assert_axioms`) and a ``test_strategy`` checking
  the strategy reaches the module's structural boxes, as its module's
  carriers are enrolled: the fast loop before the full matrix.

Properties before implementation
--------------------------------

A feature starts as mathematics, and the mathematics starts as
properties. Before implementing anything, write the laws down — on an
agent branch, as the first checkboxes of its ``TODO.md``:

1. **State the laws.** Which equations define the new structure? Which
   level of :mod:`discopy.abc` do they belong to? Which existing axioms
   must the new carrier inherit, compare :meth:`Axiom.modulo` a quotient,
   declare :meth:`Axiom.inapplicable` — or :meth:`Axiom.weaken` to a
   subspace, generating a named parameter from a membership-validating
   wrapper such as :class:`BoundaryConnected`, so that a
   :meth:`Axiom.failing` law with a green subspace shows one expected
   failure and one green cell? Write this down before any implementation.
2. **Scaffold the axioms.** Declare each law as an :class:`Axiom` on the
   abstract base class — or an ad-hoc property in its ``proptest/`` file
   when it is a boolean rather than an equation — and enrol the carrier
   in ``CARRIERS``. The body calls the operations the feature will
   provide; until they exist, the cell fails. That is the red state of
   the loop.
3. **Reach the structure.** Extend the carrier's strategy so generated
   terms actually contain the new boxes, and pin that with
   :func:`assert_strategy_finds` in the module's ``test_strategy``. A
   green cell whose strategy never generates the structure proves
   nothing.
4. **Implement until green**, on the dry run first, then the matrix.

A property is meaningful when it quantifies over all terms of a carrier.
Single behaviours — validation raises, error messages, encoding pins —
stay as unit tests in ``test/``.

Debugging a failing cell
------------------------

1. **Isolate it**: ``uv run pytest proptest/ --axioms '<carrier>.<law>'
   -x -vrsxX``. Hypothesis reports the shrunk falsifying example as
   labelled draws; on rerun the ``.hypothesis`` database replays it
   first, so the failure is stable on your machine. A failure CI found is
   in the artifact its run uploaded: with a ``GITHUB_TOKEN`` in the
   environment the ``dev`` profile reads that database too, and the cell
   fails for you the same way without a search.
2. **Record it, then debug.** DisCoPy is transparent, so the printed
   draws are valid Python building the exact counterexample. Paste them
   into a record in ``proptest/test_counterexamples.py`` (format below)
   before touching the implementation: the database remembers a failure
   only under the Hypothesis ``uv.lock`` pins and only while an artifact
   lives, while a record reproduces it on every machine, from a CI log
   included, and stays as the pin once the bug is fixed.
3. **Debug against the record**, not the search. In a REPL, call the
   record's axiom on its arguments and inspect the returned
   :class:`discopy.abc.Equation`'s sides. Do not reach for
   :meth:`Axiom.falsify` to reproduce a known failure: it searches and
   shrinks afresh each run and may land on a different counterexample, or
   none. It remains only for interactive exploration when no failure is
   in hand.
4. **Fix the root cause.** The recorded cell flips green and stays as the
   regression pin; there is nothing else to write.
5. **Or file it.** If the fix is out of scope, open an issue, declare the
   axiom ``.failing("<reason> (#<issue>)")`` where the carrier breaks it,
   and keep the record: it xfails together with the axiom, strictly, so
   the day the bug is fixed the record fails as an unexpected pass until
   the :meth:`Axiom.failing` declaration is removed — at which point the
   record is the pin. The search cell xfails too, without strictness:
   whether a search finds a rare counterexample within its budget is not
   a fact about the law.

A counterexample against an ad-hoc property that has no :class:`Axiom`
follows the same steps, except the record is a plain regression test in
the module's ``test/`` file.

Recording counterexamples
-------------------------

``proptest/test_counterexamples.py`` holds the records and their replay.
A record is structured data: the bound axiom itself and the very
arguments the search shrunk the failure to.

.. code-block:: python

    COUNTEREXAMPLES = (
        Counterexample(
            axiom=Matrix[int].copy_cocommutativity,
            args=(2, ),
            reason="Matrix.copy(x, n) is wrong for x, n >= 2 (#606)"),
        ...)

- ``axiom`` is the class attribute access, which binds the :class:`Axiom`
  to its carrier — the same object the matrix checks, so a record can
  never drift from the law it witnesses.
- ``args`` are the generated arguments, one per draw, in draw order —
  actual terms, not strings. Transparency is what lets the falsifying
  draws be pasted verbatim; their reprs are module-qualified, so extend
  the file's imports as records arrive.
- ``reason`` says what broke and links the issue when there is one.

The replay test marks a record xfail, strictly, exactly when its axiom is
declared :meth:`Axiom.failing`, and checks the equation the axiom's
:class:`AxiomFailure` carries, so the xfail is earned by the arguments
falsifying the law in one of the two shapes :meth:`Axiom.falsify` counts:
the equation is false, an assertion, or the implementation refuses to
build its terms, an :class:`discopy.utils.AxiomError`. A fixed bug shows
up as an unexpected pass, which strictness turns red, a typo'd record as
an error rather than an expected failure, and a record never needs
updating when the bug is fixed: only the ``.failing`` declaration moves.

Never delete a record because it is inconvenient; a record only leaves
when the law itself leaves the codebase.

Auditing a strategy that missed a bug
-------------------------------------

A bug found outside the matrix — by hand, by a user, in the wild — while
its law sat green is a coverage escape. The record pins the instance; the
audit closes the class. Check three causes, in order:

1. **Reach.** Can the strategy build the counterexample's shape at all?
   Ask :func:`hypothesis.find` with the carrier's strategy and a
   predicate for the shape — the structural box involved, the boundary,
   the depth. :class:`hypothesis.errors.NoSuchExample` convicts the
   strategy: extend it, then pin the reach in the module's
   ``test_strategy``, with :func:`assert_strategy_finds` when the shape is
   a box class and a bespoke ``find`` otherwise.
2. **Rarity.** Reachable but starved: run the cell with
   ``--hypothesis-show-statistics``, tagging the shape with
   :func:`hypothesis.event` if need be, to see how often it is drawn, and
   check with ``coverage run -m pytest proptest/`` that the buggy lines
   are hit at all. A shape drawn much less than once per ``max_examples``
   is invisible at the matrix's budget: rebalance the strategy's weights
   or grow its size bounds rather than raising the budget.
3. **Observation.** Drawn but not seen: the law compares its equation
   :meth:`Axiom.modulo` a quotient that erases the difference, states
   something weaker than what the bug violates, or the violated law was
   never stated — in which case the fix is a new axiom, stated first as
   in the feature protocol.

The audit is done when the search rediscovers the bug by itself: hold the
fix back and watch the cell go red without help. Only then does the suite
guard the class of bugs and not just the recorded instance.

Continuous integration
----------------------

The ``proptest`` workflow runs the suite on pull requests labelled
``proptest``, on every push to ``main``, nightly, and on manual dispatch.
``proptest/conftest.py`` registers three Hypothesis profiles, selected by
``HYPOTHESIS_PROFILE``, over one example database,
``.hypothesis/examples``:

- ``pr``, on pull requests: a small budget of new examples after the
  ``reuse`` phase has replayed every failure the database remembers, so a
  known bug fails at once and a run is fast. The workflow fixes the seed
  with ``--hypothesis-seed``, which keeps the database where
  ``derandomize`` would drop it, so a pull request draws the same
  examples every time: it is red for its own diff or for a failure the
  artifact already holds, never for luck.
- ``explore``, on ``main``, nightly and on dispatch: a large budget,
  where new counterexamples come from.
- ``dev``, the default elsewhere: a middling budget, and with a
  ``GITHUB_TOKEN`` in the environment the local database is backed by
  CI's, read-only, so what CI found replays on your machine.

Every run downloads the database the previous run uploaded as the
``hypothesis-example-db`` artifact and uploads its own afterwards,
whether or not it passed — a failed run's artifact is the one holding the
new counterexample. Hypothesis prunes what passes again and keeps what
fails, so a failure found by one night's search fails every pull request
until it is fixed or declared, with no one recording anything.

Explore runs are randomised, so a red check on ``main`` or overnight is
where a new bug surfaces: the shrunk draws in the log and the printed
``@reproduce_failure(<version>, <blob>)`` decorator reproduce it under
the Hypothesis ``uv.lock`` pins, and the artifact replays it on every
pull request and, through the ``dev`` profile, on your machine.
``--hypothesis-show-statistics`` is on, so the log of an explore run also
says how often each shape was drawn, the input of a strategy audit.
"""

from __future__ import annotations

import inspect
import pickle
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, replace
from functools import wraps
from typing import ClassVar, TypeVar, TYPE_CHECKING

from discopy.utils import (
    AxiomError, NamedGeneric, assert_iscomposable, classproperty, dumps,
    factory_name, from_tree, get_origin, loads)

if TYPE_CHECKING:
    from hypothesis import strategies as st

    from discopy import monoidal
    from discopy.abc import Equation


C0 = TypeVar("C0")
C1 = TypeVar("C1")
"""
The object and arrow types of the carrier an axiom is bound to.

An axiom annotates its arguments with these rather than with the concrete
types of the module it is written in, so that a subclass inherits the
override with its own types: :meth:`Axiom.strategy` rebinds both names to
``carrier.ob`` and ``carrier.ar`` when it evaluates the annotations. This
is also why every module stating an axiom needs
``from __future__ import annotations``, which keeps them unevaluated.
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


class AxiomFailure(AxiomError):
    """
    A law declared broken, raised when the bound axiom is called: the
    reason is the message and :attr:`equation` is the law evaluated on the
    arguments, which a recorded counterexample must falsify.
    """
    def __init__(self, reason: str, equation):
        super().__init__(reason, equation)
        self.equation = equation


@dataclass
class Axiom[T]:
    """
    A categorical law, stated either of a carrier or of one of its elements.

    An axiom whose first parameter is ``cls`` is a law of the category: it is
    bound to the carrier and its remaining arguments are generated. One whose
    first parameter is ``self`` is a law of an element, e.g. a functor, so the
    element is generated too and the law reads as a method on it.

    Calling a bound axiom returns its own verdict: :obj:`NotImplemented`
    when the structure does not apply to the carrier, and the equation
    itself otherwise; a law declared broken raises an
    :class:`AxiomFailure` carrying that equation instead of returning it.

    A law is broken when *some* argument is a counterexample, not every one,
    so :attr:`broken` is declared by :meth:`failing` before any argument is
    generated — the property matrix marks such an axiom as an expected
    failure and lets the search find the counterexample.
    """

    equation: Callable
    _: KW_ONLY
    carrier: type[T] = None
    name: str = None
    subspaces: dict = None
    broken: bool = False

    def __post_init__(self):
        if isinstance(self.equation, classmethod):
            self.equation = self.equation.__func__
        self.signature = inspect.signature(self.equation)
        self.receiver = next(iter(self.signature.parameters), None)
        self.name = self.name or self.equation.__name__
        self.subspaces = dict(self.subspaces or {})
        self.__doc__ = self.equation.__doc__

    def __repr__(self):
        """
        A bound axiom is the attribute of its carrier, e.g.
        ``cat.Arrow.unitality``; an unbound one wraps a function and has no
        transparent representation.
        """
        if self.carrier is None:
            return f"Axiom({self.name})"
        return f"{factory_name(self.carrier)}.{self.name}"

    def __hash__(self):
        return hash((self.equation, self.carrier, self.name))

    def __set_name__(self, owner, name):
        """
        Take the name of the attribute the axiom is assigned to, so that an
        override built with :meth:`modulo`, :meth:`failing` or
        :meth:`inapplicable` needs no name of its own.
        """
        self.name = name

    @property
    def is_method(self) -> bool:
        """ Whether the law is stated of an element rather than a carrier. """
        return self.receiver == "self"

    def bind(self, carrier: type[T]) -> Axiom[T]:
        """ Bind the axiom to a concrete carrier. """
        return replace(self, carrier=carrier)

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
        return replace(self, equation=equation)

    def failing(self, reason: str) -> Axiom[T]:
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

    def inapplicable(self, reason: str) -> Axiom[T]:
        """
        The same law declared not to apply to the carrier: it takes no
        argument and returns :obj:`NotImplemented`, with the reason as its
        documentation, e.g. ``trace_vanishing =
        TracedCategory.trace_vanishing.inapplicable("No trace.")``.
        """
        def law(cls):
            return NotImplemented
        law.__doc__ = reason
        return replace(self, equation=law, subspaces={}, broken=False)

    def weaken(self, **subspaces) -> Axiom[T]:
        """
        The same law quantified over a subspace of the named arguments,
        e.g. ``bifunctoriality_connected =
        MonoidalCategory.bifunctoriality.weaken(
        square=BoundaryConnected[Square[C1]])``: each named parameter
        is generated from its subspace strategy, whose wrapper validates
        membership on construction — so a recorded counterexample replays
        honestly — and is unwrapped before the body reads it. Assigned to
        its own attribute beside a ``.failing`` declaration, it shows the
        matrix one expected failure and one green cell instead of one
        blanket expected failure.
        """
        return replace(self, subspaces=dict(self.subspaces, **subspaces))

    def strategy(self) -> st.SearchStrategy:
        """
        Generate the arguments the bound axiom expects.

        ``C0`` and ``C1`` resolve to the objects and arrows of the carrier,
        or of the carrier's domain for a law of an element: the arguments a
        functor is applied to live in the category it maps from, and its
        codomain is reachable as ``self.cod`` from the body. A carrier that
        is no category, e.g. a type of objects, stands for both.

        Only the parameters' annotations are evaluated: the law's return
        annotation may name a type its module imports for checking only.
        """
        from hypothesis import strategies as st

        if self.carrier is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        function = inspect.unwrap(self.equation)
        domain = getattr(self.carrier, "dom", None)
        source = domain if self.is_method and isinstance(domain, type)\
            else self.carrier
        scope = {
            "C0": getattr(source, "ob", source),
            "C1": getattr(source, "ar", source)}
        annotations = {
            name: eval(annotation, function.__globals__, scope)
            if isinstance(annotation, str) else annotation
            for name, annotation in function.__annotations__.items()
            if name != "return"}
        annotations[self.receiver] = self.carrier
        annotations.update({
            name: substitute(annotation, scope)
            for name, annotation in self.subspaces.items()})
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

        >>> from discopy.cat import Functor
        >>> Functor.unitality.falsify()  # doctest: +ELLIPSIS
        (cat.Functor(ob_map=..., ar_map=...),)
        >>> Functor.associativity.falsify()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
         ...
        hypothesis.errors.NoSuchExample: No examples found of condition ...
        """
        from hypothesis import find

        def refutes(args):
            try:
                verdict = self(*args)
            except Exception:
                return True
            return verdict is not NotImplemented and not verdict

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
        arguments = {
            name: value.value if name in self.subspaces else value
            for name, value in bound.arguments.items()}
        if self.is_method:
            return self.equation(**arguments)
        return self.equation(
            **{self.receiver: self.carrier, **arguments})


def axiom(equation) -> Axiom:
    """ Decorate an equation as an inherited categorical axiom. """
    return Axiom(equation)


def inherited_axioms(cls) -> dict[str, Axiom]:
    """
    The axioms inherited by ``cls``, by name, subclasses overriding bases.

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


class Strategy[T](ABC):
    """
    A type with a canonical `search strategy
    <https://hypothesis.readthedocs.io/en/latest/data.html>`_
    generating its instances, and the laws every such type obeys: a term
    reads back from its representation, its pickle and its tree.
    """

    axioms = classproperty(inherited_axioms)

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
        those of the module the carrier is defined in, so that one
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
    def transparency(self) -> Equation:
        """
        The representation of a term evaluates back to it, in the
        :meth:`environment` of its type.

        The import is local because :mod:`discopy.abc` imports this module
        for its axioms, so the arrow between them cannot be reversed.
        """
        from discopy.abc import Equation

        return Equation(eval(repr(self), type(self).environment()), self)

    @axiom
    def pickling(self) -> Equation:
        """
        A term loads back from its pickle, of the same class: the equation
        is between the pairs of a class and a term, since a subscript of a
        :class:`discopy.utils.NamedGeneric` is part of what a pickle keeps.
        """
        from discopy.abc import Equation

        loaded = pickle.loads(pickle.dumps(self))
        return Equation((type(loaded), loaded), (type(self), self))

    @axiom
    def serialisation(self) -> Equation:
        """
        A term decodes back from its tree and from the JSON of its tree.
        A type without a tree declares the law inapplicable.
        """
        from discopy.abc import Equation

        return Equation(from_tree(self.to_tree()), loads(dumps(self)), self)


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

    def __repr__(self):
        return factory_name(type(self)) + f"({int(self)})"

    @classmethod
    def equation_factory(cls, *terms):
        """
        Construct an equation between natural numbers.

        The import is local because :mod:`discopy.abc` imports this module
        for its axioms, so the arrow between them cannot be reversed.
        """
        from discopy.abc import Equation

        return Equation(*terms)

    @classmethod
    def strategy(cls, *, max_size=3):
        """Generate non-negative integers."""
        from hypothesis import strategies as st

        return st.one_of(
            st.just(1),
            st.integers(min_value=0, max_value=max_size)).map(cls)

    serialisation = Strategy.serialisation.inapplicable(
        "A natural number has no tree.")


@dataclass(frozen=True)
class Atomic(Strategy, NamedGeneric["factory"]):
    """ An object of the factory containing exactly one generator. """

    value: C0

    def __post_init__(self):
        if len(self.value) != 1:
            raise ValueError("Expected an atomic object.")

    @classmethod
    def strategy(cls, **params):
        """Generate an object containing exactly one generator."""
        return resolve(cls.factory, **params).filter(
            lambda value: len(value) == 1).map(cls)


@dataclass(frozen=True)
class NonEmpty(Strategy, NamedGeneric["factory"]):
    """ A non-empty object of the factory. """

    value: C0

    def __post_init__(self):
        if not len(self.value):
            raise ValueError("Expected a non-empty object.")

    @classmethod
    def strategy(cls, **params):
        """Generate a non-empty object."""
        return resolve(cls.factory, **params).filter(bool).map(cls)


@dataclass(frozen=True)
class Subsingleton(Strategy, NamedGeneric["factory"]):
    """ An object of the factory of length at most one. """

    value: C0

    def __post_init__(self):
        if len(self.value) > 1:
            raise ValueError("Expected an object of length at most one.")

    @classmethod
    def strategy(cls, **params):
        """Generate an object of length at most one."""
        return resolve(cls.factory, **params).filter(
            lambda value: len(value) <= 1).map(cls)


@dataclass(frozen=True)
class BoundaryConnected(Strategy, NamedGeneric["factory"]):
    """
    A term whose boundary reaches every box — a hypergraph, a
    combinatorial map, or a diagram read through its map — or a pasting
    diagram of such terms, connected cell by cell.
    """

    value: C1

    def __post_init__(self):
        cells = self.value if isinstance(self.value, PastingDiagram)\
            else (self.value, )
        for cell in cells:
            graph = cell if hasattr(cell, "is_boundary_connected")\
                else cell.to_map()
            if not graph.is_boundary_connected:
                raise ValueError("Expected a boundary-connected term.")

    @classmethod
    def strategy(cls, **params):
        """Generate from the factory restricted to connected terms."""
        return resolve(
            cls.factory, boundary_connected=True, **params).map(cls)


class PastingDiagram(Strategy, NamedGeneric["factory"], tuple):
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


class ComposablePair(PastingDiagram):
    """ Two morphisms composable from left to right. """

    n_rows, n_columns = 2, 1
    n_active_rows = 2


class ComposableTriple(PastingDiagram):
    """ Three values composable from left to right. """

    n_rows, n_columns = 3, 1
    n_active_rows = 3


class HorizontalPair(PastingDiagram):
    """ Two horizontally composable cells. """

    n_rows, n_columns = 1, 2


class Square(PastingDiagram):
    """ A two-by-two grid of cells, the arguments of the interchange law. """

    n_rows = n_columns = 2


class TraceSuperposing(Strategy, NamedGeneric["factory"], tuple):
    """ A traceable arrow and an object to superpose. """

    def __new__(cls, traced: C1, obj: C0):
        traced.trace()
        return super().__new__(cls, (traced, obj))

    @classmethod
    def strategy(cls):
        """Generate a traceable identity and an arbitrary object."""
        from hypothesis import strategies as st

        object_type, arrow_type = cls.factory.ob, cls.factory
        objects = object_type.strategy()
        atomic = object_type.strategy().filter(lambda obj: len(obj) == 1)
        return st.tuples(atomic, objects).map(
            lambda pair: cls(arrow_type.id(pair[0]), pair[1]))


class TraceSliding(Strategy, NamedGeneric["factory"], tuple):
    """ Arguments for trace sliding over an arbitrary traced type. """

    left: ClassVar[bool]

    def __new__(cls, traced: C1, obj: C0, sliding: C1):
        traced_dom = obj @ sliding.cod if cls.left else sliding.cod @ obj
        traced_cod = obj @ sliding.dom if cls.left else sliding.dom @ obj
        if (traced.dom, traced.cod) != (traced_dom, traced_cod):
            raise ValueError("Expected compatible trace sliding boundaries.")
        return super().__new__(cls, (traced, obj, sliding))

    @classmethod
    def strategy(cls):
        """Generate non-trivial morphisms with compatible trace boundaries."""
        from hypothesis import strategies as st

        factory = cls.factory
        objects = factory.ob.strategy()
        traced = factory.ob.strategy(min_length=1)

        def morphisms(args):
            """ A traced morphism and one to slide, on drawn boundaries. """
            obj, dom, cod = args
            traced_dom = obj @ cod if cls.left else cod @ obj
            traced_cod = obj @ dom if cls.left else dom @ obj
            return st.tuples(
                factory.strategy(
                    dom=traced_dom, cod=traced_cod, min_leaves=1),
                factory.strategy(dom=dom, cod=cod, min_leaves=1)).map(
                    lambda pair: cls(pair[0], obj, pair[1]))

        return st.tuples(traced, objects, objects).flatmap(morphisms)


class TraceNaturalityLeft(TraceSliding):
    """ Arguments for left-oriented trace naturality. """

    left = True


class TraceNaturalityRight(TraceSliding):
    """ Arguments for right-oriented trace naturality. """

    left = False


class TraceDinaturality(Strategy, NamedGeneric["factory"], tuple):
    """
    An arrow and one to slide around its trace, traceable only once the
    sliding arrow is composed in on either side.
    """

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
    def strategy(cls):
        """Generate an arrow sliding between two traced objects."""
        from hypothesis import strategies as st

        factory = cls.factory
        objects = factory.ob.strategy()
        traced = factory.ob.strategy(min_length=1)

        def arrows(args):
            """ A traced arrow and one sliding between drawn objects. """
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


class TraceDinaturalityLeft(TraceDinaturality):
    """ Arguments for left-oriented trace dinaturality. """

    left = True


class TraceDinaturalityRight(TraceDinaturality):
    """ Arguments for right-oriented trace dinaturality. """

    left = False


class LeftCurrying(Strategy, NamedGeneric["factory"], tuple):
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
    def strategy(cls):
        """Generate an evaluation suitable for left or right currying."""
        from hypothesis import strategies as st

        object_type, arrow_type = cls.factory.ob, cls.factory
        objects = object_type.strategy().filter(lambda obj: len(obj) == 1)
        return st.tuples(objects, objects).map(lambda pair: cls(
            arrow_type.ev(*pair, left=cls.left), *pair))


class RightCurrying(LeftCurrying):
    """ Arguments for right currying followed by evaluation. """

    left = False


class FeedbackVanishing(Strategy, NamedGeneric["factory"], tuple):
    """ A feedback arrow together with the monoidal unit. """

    def __new__(cls, arrow: C1, unit: C0):
        if len(unit):
            raise ValueError("Expected the monoidal unit.")
        arrow.feedback(mem=unit)
        return super().__new__(cls, (arrow, unit))

    @classmethod
    def strategy(cls, **params):
        """Generate a feedback arrow paired with the monoidal unit."""
        object_type, arrow_type = cls.factory.ob, cls.factory
        return arrow_type.strategy(**params).map(
            lambda arrow: cls(arrow, object_type()))


class FeedbackJoining(Strategy, NamedGeneric["factory"], tuple):
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
    def strategy(cls):
        """Generate a feedback arrow with two units of memory."""
        from hypothesis import strategies as st

        object_type, arrow_type = cls.factory.ob, cls.factory
        objects = object_type.strategy()
        atomic = object_type.strategy().filter(lambda obj: len(obj) == 1)

        def arrows(args):
            """ A feedback arrow over a drawn object and two memory units. """
            obj, first, second = args
            memory = first @ second
            return arrow_type.strategy(
                dom=obj @ memory.delay(), cod=obj @ memory).map(
                    lambda arrow: cls(arrow, memory))

        return st.tuples(objects, atomic, atomic).flatmap(arrows)


class HomogeneousMemory(FeedbackJoining):
    """ A feedback arrow whose units of memory are all the same object. """

    def __new__(cls, arrow: C1, memory: C0):
        if any(memory[i:i + 1] != memory[:1] for i in range(len(memory))):
            raise ValueError("Expected homogeneous memory.")
        return super().__new__(cls, arrow, memory)

    @classmethod
    def strategy(cls):
        """Generate a feedback arrow with two units of the same memory."""
        from hypothesis import strategies as st

        object_type, arrow_type = cls.factory.ob, cls.factory
        objects = object_type.strategy()
        atomic = object_type.strategy().filter(lambda obj: len(obj) == 1)

        def arrows(args):
            """ A feedback arrow over a drawn object and a doubled atom. """
            obj, atom = args
            memory = atom @ atom
            return arrow_type.strategy(
                dom=obj @ memory.delay(), cod=obj @ memory).map(
                    lambda arrow: cls(arrow, memory))

        return st.tuples(objects, atomic).flatmap(arrows)


@dataclass(frozen=True, eq=False)
class Relabelling(Mapping):
    """
    A map on the generators of a free category, sending the atoms it names to
    a chosen object, every other atom to itself, and a box to one of the
    same name on the relabelled boundary.

    It is a :class:`Mapping` rather than a closure so that functors built
    from it can be composed and compared, which is what makes the axioms of
    ``Cat`` itself checkable: :meth:`discopy.utils.MappingOrCallable.then`
    composes by iterating the keys of the left-hand map, and equality
    compares the wrapped maps. Iterating yields only the atoms it renames,
    while looking up is total, so a functor built from it as both its
    object and its arrow map applies to any diagram and still composes to
    something comparable.
    """
    images: tuple[tuple[object, object], ...] = ()

    def __repr__(self):
        return factory_name(type(self)) + f"(images={self.images!r})"

    def __getitem__(self, key):
        """
        The image of an atom, looked up by name, or of a box, relabelled on
        its boundary by the functor of the box's own category: a rotation or
        a delay of an atom is that functor's business, e.g.
        :class:`discopy.rigid.Functor`'s, on a box's boundary as on an
        object.

        The import is local because :mod:`discopy.cat` imports this module
        for its strategies, so the arrow between them cannot be reversed.
        """
        from discopy.cat import Functor, Ob

        if not isinstance(key, Ob):
            functor = getattr(type(key), "functor_factory", Functor)
            relabel = functor(self, self)
            return type(key)(key.name, relabel(key.dom), relabel(key.cod))
        wire, = getattr(key, "inside", (key, ))
        for atom, image in self.images:
            other, = getattr(atom, "inside", (atom, ))
            if other.name == wire.name:
                return image
        return key

    def __iter__(self):
        return iter([atom for atom, _ in self.images])

    def __len__(self):
        return len(self.images)

    def __bool__(self):
        """ A relabelling is total, even when it renames nothing. """
        return True


def resolve(annotation, **params) -> st.SearchStrategy:
    """ Resolve the strategy implemented by an annotated type. """
    if not isinstance(annotation, type)\
            or not issubclass(annotation, Strategy):
        raise TypeError(
            f"Expected a Strategy annotation, got {annotation!r}.")
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


def assert_axioms(*carriers) -> None:
    """
    Check every axiom of each carrier on a single generated example, a dry
    run of the property tests in ``proptest/``.

    An axiom that does not apply is skipped, a broken one is only required
    to raise its :class:`discopy.utils.AxiomError` — one example need not
    be a counterexample — and any other law must hold.
    """
    from hypothesis import Phase, find, settings

    single_shot = settings(
        max_examples=1, phases=(Phase.generate, ), database=None)
    for carrier in carriers:
        for axiom in carrier.axioms.values():
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


def assert_strategy_finds(
        carrier: type[monoidal.Diagram], *structures: type) -> None:
    """
    Check that the strategy of a diagram carrier generates a term
    containing a box of each of the given structural classes.
    """
    from hypothesis import find

    for structure in structures:
        find(carrier.strategy(), lambda term: any(
            isinstance(box, structure) for box in term.boxes))
