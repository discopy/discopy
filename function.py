# -*- coding: utf-8 -*-
"""
Implements the monoidal category of functions on lists
with direct sum as tensor.

>>> @discofunc(Dim(2), Dim(2))
... def swap(x):
...     return x[::-1]
>>> assert isinstance(swap, Function)
>>> assert (swap >> swap)([1, 2]) == [1, 2]
>>> assert (swap @ swap)([0, 1, 2, 3]) == [1, 0, 3, 2]

Copy and add form a bimonoid.

>>> @discofunc(Dim(1), Dim(2))
... def copy(x):
...     return x + x
>>> assert copy([1]) == [1, 1]
>>> @discofunc(Dim(2), Dim(1))
... def add(x):
...     return [x[0] + x[1]]
>>> assert add([1, 2]) == [3]
>>> assert (copy @ copy >> Id(1) @ swap @ Id(1) >> add @ add)([123, 25]) ==\\
...        (add >> copy)([123, 25])
"""
from functools import reduce as fold
from discopy import cat, config
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor


if config.JAX:
    import warnings
    for msg in config.IGNORE:
        warnings.filterwarnings("ignore", message=msg)
    import jax.numpy as np
else:
    import numpy as np


class Dim(Ty):
    """ Implements dimensions as tuples of non-negative integers.
    Dimensions form a monoid with product @ and unit Dim(0).

    >>> Dim(0) @ Dim(1) @ Dim(2)
    Dim(3)
    """
    def __init__(self, x):
        """
        >>> Dim(-1)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected non-negative integer, got -1 instead.
        """
        if not isinstance(x, int) or x < 0:
            raise ValueError("Expected non-negative integer, "
                             "got {} instead.".format(repr(x)))
        self._dim = x
        super().__init__(*[Ob(1) for i in range(x)])

    @property
    def dim(self):
        """
        >>> assert Dim(5).dim == 5
        """
        return self._dim

    def __matmul__(self, other):
        """
        >>> assert Dim(0) @ Dim(4) == Dim(4) @ Dim(0) == Dim(4)
        >>> assert Dim(2) @ Dim(3) == Dim(5)
        """
        return Dim(self.dim + other.dim)

    def __add__(self, other):
        """
        >>> assert sum([Dim(0), Dim(3), Dim(4)], Dim(0)) == Dim(7)
        """
        return self @ other

    def __repr__(self):
        """
        >>> Dim(5)
        Dim(5)
        """
        return "Dim({})".format(self.dim)

    def __str__(self):
        """
        >>> print(Dim(0) @ Dim(3))
        Dim(3)
        """
        return repr(self)

    def __hash__(self):
        """
        >>> dim = Dim(3)
        >>> {dim: 42}[dim]
        42
        """
        return hash(repr(self))


class Function(Box):
    """ Wraps functions on lists with domain and codomain information
    """
    def __init__(self, name, dom, cod, function):
        """
        >>> f = Function('f', Dim(2), Dim(2), lambda x: x)
        """
        if not isinstance(dom, Dim):
            raise ValueError(
                "Dim expected for name, got {} instead.".format(type(dom)))
        if not isinstance(cod, Dim):
            raise ValueError(
                "Dim expected for name, got {} instead.".format(type(cod)))
        self._function = function
        if not isinstance(name, str):
            raise ValueError(
                "String expected for name, got {} instead.".format(type(name)))
        self._name = name
        super().__init__(name, dom, cod)

    @property
    def function(self):
        return self._function

    @property
    def name(self):
        return self._name

    def __repr__(self):
        """
        >>> Function('Id_2', Dim(2), Dim(2), lambda x: x)
        Id_2
        """
        return self.name

    def __str__(self):
        """
        >>> print(Function('copy', Dim(2), Dim(2), lambda x: x + x))
        copy
        """
        return repr(self)

    def __call__(self, value):
        """
        >>> f = Function('f', Dim(2), Dim(2), lambda x: x)
        >>> assert f([1, 2]) == [1, 2]
        >>> f(465)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: List expected, got <class 'int'> instead.
        >>> f([3])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        function.AxiomError: Expected input of length 2, got 1 instead.
        """
        if not isinstance(value, list):
            raise ValueError(
                "List expected, got {} instead.".format(type(value)))
        if not len(value) == self.dom.dim:
            raise AxiomError("Expected input of length {}, got {} instead."
                             .format(self.dom.dim, len(value)))
        return self.function(value)

    def then(self, other):
        """
        >>> inv = Function('inv', Dim(2), Dim(2), lambda x: x[::-1])
        >>> inv >> inv
        inv >> inv
        >>> assert (inv >> inv)([1, 0]) == [1, 0]
        """
        if not isinstance(other, Function):
            raise ValueError(
                "Function expected, got {} instead.".format(repr(other)))
        if self.cod.dim != other.dom.dim:
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        newname = self.name + ' >> ' + other.name

        def f(x):
            return other(self(x))
        return Function(newname, self.dom, other.cod, f)

    def tensor(self, other):
        """
        >>> add = Function('add', Dim(2), Dim(1), lambda x: [x[0] + x[1]])
        >>> copy = Function('copy', Dim(1), Dim(2), lambda x: x + x)
        >>> assert (add @ copy)([0, 1, 2]) == [1, 2, 2]
        >>> add @ copy
        add @ copy
        """
        if not isinstance(other, Function):
            raise ValueError(
                "Function expected, got {} instead.".format(repr(other)))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        newname = self.name + ' @ ' + other.name

        def f(x):
            return self(x[:self.dom.dim]) + other(x[self.dom.dim:])
        return Function(newname, dom, cod, f)


class AxiomError(cat.AxiomError):
    """
    """


class Id(Function):
    """ Implements the identity function for a given dimension.

    >>> Id(5)
    Id(5)
    >>> assert Id(1)([476]) == [476]
    >>> assert Id(2)([0, 1]) == [0, 1]
    """
    def __init__(self, dim):
        name = 'Id({})'.format(dim)
        super().__init__(name, Dim(dim), Dim(dim), lambda x: x)


def discofunc(dom, cod):
    """
    Decorator turning a python function into a discopy Function
    with domain and codomain information.

    >>> @discofunc(Dim(2), Dim(2))
    ... def f(x):
    ...     return x[::-1]
    >>> assert isinstance(f, Function)
    """
    def decorator(f):
        return Function(f.__name__, dom, cod, f)
    return decorator
