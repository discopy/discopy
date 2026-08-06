"""Data structures and strategies for property tests."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

from discopy.utils import assert_iscomposable

if TYPE_CHECKING:
    from hypothesis import strategies as st


class Strategy[T](ABC):
    """A type with a canonical property-test strategy."""

    @classmethod
    @abstractmethod
    def strategy(cls, **params) -> "st.SearchStrategy[T]":
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
    """ Arguments satisfying the trace sliding boundaries. """

    def __new__(cls, traced: C1, obj: C0, sliding: C1):
        if sliding.dom != obj or sliding.cod != obj:
            raise ValueError("Expected an endomorphism on the sliding object.")
        if traced.dom != obj @ obj or traced.cod != obj @ obj:
            raise ValueError("Expected an endomorphism on two copies.")
        return super().__new__(cls, (traced, obj, sliding))

    @classmethod
    def strategy(cls, *, factory: type[C1]):
        """Generate identities satisfying the trace sliding boundaries."""
        object_type, arrow_type = factory.ob, factory
        return object_type.strategy().filter(lambda obj: len(obj) == 1).map(
            lambda obj: cls(
                arrow_type.id(obj @ obj), obj, arrow_type.id(obj)))


class LeftCurrying[C0, C1](
        Strategy[tuple[C1, C0, C0]], tuple[C1, C0, C0]):
    """ Arguments for left currying followed by evaluation. """

    left = True

    def __new__(cls, arrow: C1, base: C0, exponent: C0):
        arrow.curry(left=cls.left)
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
    """ A feedback arrow with a non-empty memory object. """

    def __new__(cls, arrow: C1, memory: C0):
        if not len(memory):
            raise ValueError("Expected a non-empty memory object.")
        arrow.feedback(mem=memory)
        arrow.feedback().feedback()
        return super().__new__(cls, (arrow, memory))

    @classmethod
    def strategy(cls, *, factory: type[C1]):
        """Generate a feedback arrow with two units of memory."""
        from hypothesis import strategies as st

        object_type, arrow_type = factory.ob, factory
        objects = object_type.strategy()
        atomic = object_type.strategy().filter(lambda obj: len(obj) == 1)

        def arrows(args):
            obj, atom = args
            memory = atom @ atom
            return arrow_type.strategy(
                dom=obj @ memory.delay(), cod=obj @ memory).map(
                    lambda arrow: cls(arrow, memory))

        return st.tuples(objects, atomic).flatmap(arrows)


class Axiom[T]:
    """ A carrier-parametrised equation with explicit arguments. """

    def __init__(self, equation, *, strict=True, carrier=None):
        self.equation = equation if isinstance(equation, classmethod)\
            else classmethod(equation)
        function = self.equation.__func__
        self.signature = inspect.signature(function)
        self.carrier = carrier
        self.name = self.__name__ = function.__name__
        self.strict = strict
        self.__doc__ = function.__doc__

    def __repr__(self):
        return f"Axiom({self.name})"

    def bind(self, carrier: type[T]) -> Axiom[T]:
        """ Bind the axiom to a concrete carrier. """
        return type(self)(
            self.equation, strict=self.strict, carrier=carrier)

    def __get__(self, instance, owner: type[T]) -> Axiom[T]:
        return self.bind(owner)

    @property
    def parameters(self) -> tuple[inspect.Parameter, ...]:
        """ The explicit parameters of the equation. """
        return tuple(self.signature.parameters.values())[1:]

    def __call__(self, *args, **kwargs):
        if self.carrier is None:
            raise TypeError(f"{self.name} is not bound to a class.")
        signature = self.signature.replace(parameters=self.parameters)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = {
            next(iter(self.signature.parameters)): self.carrier,
            **bound.arguments}
        result = self.equation.__func__(**arguments)
        equation_type = type(self.carrier.equation_factory())
        if not isinstance(result, equation_type):
            raise TypeError(
                f"{self.name} returned {type(result).__name__}, "
                f"expected an {equation_type}.")
        return result


def axiom(equation=None, *, strict=True):
    """ Decorate an equation as an inherited categorical axiom. """
    if equation is None:
        return lambda function: Axiom(function, strict=strict)
    return Axiom(equation, strict=strict)
