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

Copy and add form a bialgebra.

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
    Dim(1, 2)
    """
    def __init__(self, *xs):
        """
        >>> len(Dim(2, 3, 4))
        3
        >>> len(Dim(1, 2, 3))
        3
        >>> Dim(-1)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected non-negative integer, got -1 instead.
        """
        for x in xs:  # pylint: disable=invalid-name
            if not isinstance(x, int) or x < 0:
                raise ValueError("Expected non-negative integer, "
                                 "got {} instead.".format(repr(x)))
        self._dim = sum(xs)
        super().__init__(*[Ob(x) for x in xs if x > 0])

    @property
    def dim(self):
        """
        >>> assert Dim(2, 3).dim == 5
        """
        return self._dim

    def __matmul__(self, other):
        """
        >>> assert Dim(0) @ Dim(2, 3) == Dim(2, 3) @ Dim(0) == Dim(2, 3)
        """
        return Dim(*[x.name for x in super().__matmul__(other)])

    def __add__(self, other):
        """
        >>> assert sum([Dim(0), Dim(2, 3), Dim(4)], Dim(0)) == Dim(2, 3, 4)
        """
        return self @ other

    def __getitem__(self, key):
        """
        >>> assert Dim(2, 3)[:1] == Dim(3, 2)[1:] == Dim(2)
        >>> assert Dim(2, 3)[0] == Dim(3, 2)[1] == 2
        """
        if isinstance(key, slice):
            return Dim(*[x.name for x in super().__getitem__(key)])
        return super().__getitem__(key).name

    def __iter__(self):
        """
        >>> [n for n in Dim(2, 3, 4)]
        [2, 3, 4]
        """
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        """
        >>> Dim(0, 1, 2)
        Dim(1, 2)
        """
        return "Dim({})".format(', '.join(map(repr, self)) or '0')

    def __str__(self):
        """
        >>> print(Dim(0, 1, 2))
        Dim(1, 2)
        """
        return repr(self)

    def __hash__(self):
        """
        >>> dim = Dim(2, 3)
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
        dom, cod = Dim(*dom), Dim(*cod)
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

    >>> Id(2, 3)
    Id(2, 3)
    >>> assert Id(1)([476]) == [476]
    >>> assert Id(2)([0, 1]) == [0, 1]
    """
    def __init__(self, *dim):
        dims = dim[0] if isinstance(dim[0], Dim) else Dim(*dim)
        name = 'Id({})'.format(', '.join(map(repr, dim)))
        super().__init__(name, dims, dims, lambda x: x)


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
