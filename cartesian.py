# -*- coding: utf-8 -*-
"""
Implements functors into the category of functions on tuples
with cartesian product as tensor.

The cartesian product comes with swap, copy and discard maps:

>>> COPY = Box('copy', 1, 2, lambda *x: x + x)
>>> SWAP = Box('swap', 2, 2, lambda x, y: (y, x))
>>> DISCARD = Box('discard', 1, 0, lambda *x: ())

Comonoid law:

>>> assert (COPY >> Id(1) @ DISCARD)(42) == Id(1)(42)\\
...     == (COPY >> DISCARD @ Id(1))(42)

Yang-Baxter equation:

>>> assert (SWAP @ Id(1) >> Id(1) @ SWAP >> SWAP @ Id(1))(1, 2, 3)\\
...     == (Id(1) @ SWAP >> SWAP @ Id(1) >> Id(1) @ SWAP)(1, 2, 3)

Bialgebra law:

>>> assert (ADD >> COPY)(1, 2)\\
...     == (COPY @ COPY >> Id(1) @ SWAP @ Id(1) >> ADD @ ADD)(1, 2)

Naturality of the symmetry:

>>> f = disco(1, 1)(lambda x: x + 1)
>>> g = disco(1, 1)(lambda x: 2 * x)
>>> assert (f @ g >> SWAP)(42, 34) == (SWAP >> g @ f)(42, 34)
"""

from discopy.cat import AxiomError
from discopy import messages, rigidcat
from discopy.cat import Quiver
from discopy.rigidcat import PRO, RigidFunctor


def tuplify(xs):
    return xs if isinstance(xs, tuple) else (xs, )


def untuplify(*xs):
    return xs[0] if len(xs) == 1 else xs


class Function(rigidcat.Box):
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

    >>> sort = Function(3, 3, lambda *xs: tuple(sorted(xs)))
    >>> swap = Function(2, 2, lambda x, y: (y, x))
    >>> assert (sort >> Function.id(1) @ swap)(1, 2, 3) == (1, 3, 2)
    """
    def __init__(self, dom, cod, function):
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
        return Function(self.dom, other.cod,
                        lambda *vals: other(*tuplify(self(*vals))))

    def tensor(self, other):
        """
        Returns the parallel composition of 'self' and 'other'.

        >>> copy = Function(1, 2, lambda *x: x + x)
        >>> swap = Function(2, 2, lambda x, y: (y, x))
        >>> assert (swap @ swap)(1, 2, 3, 4) == (2, 1, 4, 3)
        >>> assert (copy @ copy)(1, 2) == (1, 1, 2, 2)
        """
        if not isinstance(other, Function):
            raise TypeError(messages.type_err(Function, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod

        def product(*vals):
            vals0 = tuplify(self(*vals[:len(self.dom)]))
            vals1 = tuplify(other(*vals[len(self.dom):]))
            return untuplify(*(vals0 + vals1))
        return Function(dom, cod, product)

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
        return Function(dom, dom, untuplify)


class PythonFunctor(RigidFunctor):
    """
    Implements functors into the category of Python functions on tuples
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_cls=PRO, ar_cls=Function)


class Diagram(rigidcat.Diagram):
    """
    Implements diagrams of Python functions.
    """
    def __init__(self, dom, cod, boxes, offsets, layers=None):
        super().__init__(PRO(dom), PRO(cod), boxes, offsets, layers=layers)

    @staticmethod
    def _upgrade(diagram):
        """
        Takes a rigidcat.Diagram and returns a cartesian.Diagram.
        """
        return Diagram(len(diagram.dom), len(diagram.cod),
                       diagram.boxes, diagram.offsets, layers=diagram.layers)

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
        ob = Quiver(lambda t: PRO(len(t)))
        ar = Quiver(lambda f:
                    Function(len(f.dom), len(f.cod), f.function))
        return PythonFunctor(ob, ar)(self)(*values)


class Id(Diagram):
    """
    Implements identity diagrams on dom inputs.

    >>> c =  SWAP >> ADD >> COPY
    >>> assert Id(2) >> c == c == c >> Id(2)
    """
    def __init__(self, dom):
        """
        >>> assert Diagram.id(42) == Id(42) == Diagram(42, 42, [], [])
        """
        super().__init__(PRO(dom), PRO(dom), [], [], layers=None)

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


class Box(rigidcat.Box, Diagram):
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
        rigidcat.Box.__init__(self, name, PRO(dom), PRO(cod), data=data)
        Diagram.__init__(self, dom, cod, [self], [0])

    @property
    def function(self):
        return self._function

    def __repr__(self):
        return "Box({}, {}, {}{}{})".format(
            repr(self.name), len(self.dom), len(self.cod),
            ', function=' + repr(self.function) if self.function else '',
            ', data=' + repr(self.data) if self.data else '')


class CartesianFunctor(RigidFunctor):
    """
    Implements functors into the category of Python functions on tuples.

    >>> x = rigidcat.Ty('x')
    >>> f, g = rigidcat.Box('f', x, x @ x), rigidcat.Box('g', x @ x, x)
    >>> ob = {x: PRO(1)}
    >>> ar = {f: COPY, g: ADD}
    >>> F = CartesianFunctor(ob, ar)
    >>> assert F(f >> g)(43) == 86
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
    >>> copy = disco(1, 2, name='copy')(lambda x: (x, x))
    """
    def decorator(func):
        if name is None:
            return Box(func.__name__, dom, cod, func)
        return Box(name, dom, cod, func)
    return decorator


COPY = Box('copy', 1, 2, lambda *x: x + x)
SWAP = Box('swap', 2, 2, lambda x, y: (y, x))
DISCARD = Box('discard', 1, 0, lambda *x: ())
ADD = Box('add', 2, 1, lambda x, y: x + y)
