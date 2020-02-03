# -*- coding: utf-8 -*-
"""
Implements functors into the category of functions on tuples
with cartesian product as tensor.

>>> assert (ADD >> COPY)(1, 2)\\
...     == (COPY @ COPY >> Id(1) @ SWAP @ Id(1) >> ADD @ ADD)(1, 2)
"""

from discopy.cat import AxiomError
from discopy import messages, moncat
from discopy.cat import Quiver
from discopy.moncat import PRO, MonoidalFunctor


class Function(moncat.Box):
    """
    Wraps python functions with domain and codomain information.

    Parameters
    ----------
    dom : int
        Domain of the diagram.
    cod : int
        Codomain of the diagram.
    function: any
        Python function with a call method.
    """
    def __init__(self, dom, cod, function):
        if isinstance(dom, PRO):
            dom = len(dom)
        if isinstance(cod, PRO):
            cod = len(cod)
        if not isinstance(dom, int):
            raise TypeError(messages.type_err(int, dom))
        if not isinstance(cod, int):
            raise TypeError(messages.type_err(int, cod))
        self._function = function
        super().__init__(repr(function), PRO(dom), PRO(cod))

    @property
    def function(self):
        """
        The function stored in a discopy.Function object is immutable

        >>> f = Function(2, 2, lambda x: x)
        >>> f.function = lambda x: 2*x  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._function

    def __repr__(self):
        return "Function(dom={}, cod={}, function={})".format(
            self.dom, self.cod, repr(self.function))

    def __str__(self):
        return repr(self)

    def __call__(self, *values):
        """
        In order to call a Function, it is sufficient that the input
        has a length which agrees with the domain dimension.

        Parameters
        ----------
        values : tuple
        """
        if not len(values) == len(self.dom):
            raise AxiomError("Expected input of length {}, got {} instead."
                             .format(len(self.dom), len(values)))
        return self.function(*values)

    def then(self, other):
        """
        Returns the sequential composition of 'self' with 'other'.
        This method is called using the binary operators `>>` and `<<`.

        >>> copy = Function(1, 2, lambda *x: x + x)
        >>> swap = Function(2, 2, lambda x, y: (y, x))
        >>> assert (copy >> swap)(1) == copy(1)
        >>> assert (swap >> swap)(1, 2) == (1, 2)
        """
        if not isinstance(other, Function):
            raise TypeError(messages.type_err(Function, other))
        if len(self.cod) != len(other.dom):
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))

        def func(*vals):
            return other(*self(*vals))
        return Function(self.dom, other.cod, func)

    def tensor(self, other):
        """
        Returns the parallel composition of 'self' and 'other'.
        This method is called using the binary operator `@`.

        >>> copy = Function(1, 2, lambda *x: x + x)
        >>> swap = Function(2, 2, lambda x, y: (y, x))
        >>> assert (swap @ swap)(1, 2, 3, 4) == (2, 1, 4, 3)
        >>> assert (copy @ copy)(1, 2) == (1, 1, 2, 2)
        """
        if not isinstance(other, Function):
            raise TypeError(messages.type_err(Function, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod

        def func(*vals):
            vals0 = vals[:len(self.dom)]
            vals1 = vals[len(self.dom):]
            return self(*vals0) + other(*vals1)
        return Function(dom, cod, func)

    @staticmethod
    def id(dom):
        """
        >>> assert Function.id(0)() == ()
        >>> assert Function.id(2)(1, 2) == (1, 2)
        >>> Function.id(1)(1, 2)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        discopy.cat.AxiomError: Expected input of length 1, got 2 instead.
        """
        return Function(dom, dom, lambda *xs: xs)


class PythonFunctor(MonoidalFunctor):
    """
    Implements functors into the category of Python functions on tuples
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_cls=PRO, ar_cls=Function)


class Diagram(moncat.Diagram):
    """
    Implements diagrams of Python functions.
    """
    def __init__(self, dom, cod, boxes, offsets):
        super().__init__(PRO(dom), PRO(cod), boxes, offsets)

    def then(self, other):
        """
        >>> assert isinstance(ADD >> COPY, Diagram)
        """
        result = super().then(other)
        return Diagram(len(result.dom), len(result.cod),
                       result.boxes, result.offsets)

    def tensor(self, other):
        """
        >>> assert (ADD @ ADD >> Id(1) @ COPY).offsets == [0, 1, 1]
        """
        result = super().tensor(other)
        return Diagram(len(result.dom), len(result.cod),
                       result.boxes, result.offsets)

    @staticmethod
    def id(x):
        """
        >>> Diagram.id(2)
        Id(2)
        """
        return Id(x)

    def __call__(self, *values):
        """
        >>> assert SWAP(1, 2) == (2, 1)
        >>> assert (COPY @ COPY >> Id(1) @ SWAP @ Id(1))(1, 2) == (1, 2, 1, 2)
        """
        ob = Quiver(lambda t: t)
        ar = Quiver(lambda f:
                    Function(f.dom, f.cod, f.function))
        return PythonFunctor(ob, ar)(self)(*values)


class Id(Diagram):
    """ Implements identity diagrams on n inputs.

    >>> c =  SWAP >> ADD >> COPY
    >>> assert Id(2) >> c == c == c >> Id(2)
    """
    def __init__(self, dim):
        """
        >>> assert Diagram.id(42) == Id(42) == Diagram(42, 42, [], [])
        """
        if isinstance(dim, PRO):
            dim = len(dim)
        super().__init__(dim, dim, [], [])

    def __repr__(self):
        """
        >>> Id(42)
        Id(42)
        """
        return "Id({})".format(len(self.dom))

    def __str__(self):
        """
        >>> print(Id(42))
        Id(42)
        """
        return repr(self)


class Box(moncat.Box, Diagram):
    """
    Implements Python functions as boxes in a learner.Diagram.
    """
    def __init__(self, name, dom, cod, function=None, data=None):
        """
        >>> assert COPY.dom == PRO(1)
        >>> assert COPY.cod == PRO(2)
        """
        if function is not None:
            self._function = function
        moncat.Box.__init__(self, name, PRO(dom), PRO(cod), data=data)
        Diagram.__init__(self, dom, cod, [self], [0])

    @property
    def function(self):
        return self._function

    def __repr__(self):
        return "Box({}, {}, {}{}{})".format(
            repr(self.name), len(self.dom), len(self.cod),
            ', function=' + repr(self.function) if self.function else '',
            ', data=' + repr(self.data) if self.data else '')


class CartesianFunctor(MonoidalFunctor):
    """
    Implements functors into the category of Python functions on tuples.
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_cls=PRO, ar_cls=Diagram)


def disco(dom, cod, name=None):
    """
    Decorator turning a python function into a cartesian.Box storing it,
    given domain and codomain information.

    >>> @disco(2, 1)
    ... def add(x, y):
    ...     return (x + y,)
    >>> assert isinstance(add, Box)
    """
    def decorator(func):
        if name is None:
            return Box(func.__name__, dom, cod, func)
        return Box(name, dom, cod, func)
    return decorator


COPY = Box('copy', 1, 2, lambda *x: x + x)
SWAP = Box('swap', 2, 2, lambda x, y: (y, x))
DISCARD = Box('discard', 1, 0, lambda *x: ())
ADD = Box('add', 2, 1, lambda x, y: (x + y, ))
