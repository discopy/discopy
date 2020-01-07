# -*- coding: utf-8 -*-
"""
Implements the symmetric monoidal category (PROP) of functions on Numpy lists
with cartesian product as tensor.

Symmetry.

>>> @discofunc(2, 2)
... def swap(x):
...     return x[::-1]
>>> assert isinstance(swap, Function)
>>> assert np.all((swap >> swap)([1, 2]) == np.array([1, 2]))
>>> assert np.all((swap @ swap)([0, 1, 2, 3]) == np.array([1, 0, 3, 2]))

Dim(0) is a terminal object with the following discarding map.

>>> discard = lambda n: Function('discard', Dim(n), Dim(0), lambda x: [])
>>> assert discard(3)([23, 2, 67]) == [] == discard(1)([23])

Copy and add form a bimonoid.

>>> @discofunc(1, 2)
... def copy(x):
...     return np.concatenate([x, x])
>>> assert np.all(copy([1]) == np.array([1, 1]))
>>> @discofunc(2, 1)
... def add(x):
...     return np.array([x[0] + x[1]])
>>> assert np.all(add([1, 2]) == np.array([3]))
>>> assert np.all((copy @ copy >> Id(1) @ swap @ Id(1) >>\\
...               add @ add)([123, 25]) == (add >> copy)([123, 25]))

Numpy/Jax functions are also accepted.

>>> abs = Function('abs', Dim(2), Dim(2), np.absolute)
>>> assert np.all((swap >> abs)(np.array([-1, -2])) == np.array([2, 1]))
>>> def scalar_mult(scalar):
...     return Function(repr(scalar), Dim(1), Dim(1), lambda x: scalar * x)
>>> assert np.all((scalar_mult(2) @ scalar_mult(1) >> abs)(np.array([-1, -2]))
...               == np.array([2, 2]))
"""
from functools import reduce as fold
from discopy import cat
from discopy.matrix import np
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor


class Dim(Ty):
    """ Implements dimensions as non-negative integers.
    These form a monoid with sum as product denoted by @ and unit Dim(0).

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
    """
    Wraps python functions with domain and codomain information.

    >>> swap = Function('swap', Dim(2), Dim(2), lambda x: x[::-1])
    >>> assert np.all(swap([1, 2]) == np.array([2, 1]))
    >>> assert np.all(swap(np.array([1, 2])) == np.array([2, 1]))
    >>> abs = Function('abs', Dim(2), Dim(2), np.absolute)
    >>> assert np.all(abs(np.array([1, -1])) == np.array([1, 1]))
    >>> assert np.all((swap >> abs)(np.array([-1, -2])) == np.array([2, 1]))
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
        """
        >>> assert Function('Id', Dim(2), Dim(2), lambda x: x).function(1) == 1
        """
        return self._function

    @property
    def name(self):
        """
        The name of a function is immutable.

        >>> f = Function('f', Dim(2), Dim(2), lambda x: x)
        >>> assert f.name == 'f'
        >>> f.name = 'g'  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._name

    def __repr__(self):
        """
        >>> id = Function('Id_2', Dim(2), Dim(2), lambda x: x)
        >>> assert 'Function(name=Id_2, dom=Dim(2), cod=Dim(2)' in repr(id)
        """
        return "Function(name={}, dom={}, cod={}, function={})".format(
            self.name, self.dom, self.cod, repr(self.function))

    def __str__(self):
        """
        >>> print(Function('copy', Dim(2), Dim(2), lambda x: x + x))
        copy
        """
        return self.name

    def __call__(self, value):
        """
        In order to call a Function, it is sufficient that the input
        has a length which agrees with the domain dimension.

        >>> f = Function('f', Dim(2), Dim(2), lambda x: x)
        >>> assert np.all(f([1, 2]) == np.array([1, 2]))
        >>> f([3])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        function.AxiomError: Expected input of length 2, got 1 instead.
        """
        if isinstance(value, list):
            value = np.array(value)
        if not len(value) == self.dom.dim:
            raise AxiomError("Expected input of length {}, got {} instead."
                             .format(self.dom.dim, len(value)))
        return self.function(value)

    def then(self, other):
        """
        >>> swap = Function('swap', Dim(2), Dim(2), lambda x: x[::-1])
        >>> print(swap >> swap)
        (swap >> swap)
        >>> assert np.all((swap >> swap)([1, 0]) == np.array([1, 0]))
        >>> id = Function('id', Dim(3), Dim(3), lambda x: x)
        >>> id >> swap  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        function.AxiomError: Function(... does not compose with Function(...
        """
        if not isinstance(other, Function):
            raise ValueError(
                "Function expected, got {} instead.".format(repr(other)))
        if self.cod.dim != other.dom.dim:
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        newname = '(' + self.name + ' >> ' + other.name + ')'

        def f(x):
            return other(self(x))
        return Function(newname, self.dom, other.cod, f)

    def tensor(self, other):
        """
        >>> add = Function('add', Dim(2), Dim(1),
        ...                lambda x: np.array([x[0] + x[1]]))
        >>> copy = Function('copy', Dim(1), Dim(2),\\
        ...                 lambda x: np.concatenate([x, x]))
        >>> assert np.all((add @ copy)([3, 1, 2]) == np.array([4, 2, 2]))
        >>> assert (add @ copy @ Id(0)).name == '(add @ copy)'
        >>> print(Id(0) @ Id(0))
        Id(0)
        """
        if not isinstance(other, Function):
            raise ValueError(
                "Function expected, got {} instead.".format(repr(other)))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        if not self.name == 'Id(0)':
            if not other.name == 'Id(0)':
                newname = '(' + self.name + ' @ ' + other.name + ')'
            else:
                newname = self.name
        else:
            newname = other.name

        def f(x):
            return np.concatenate([self(x[:self.dom.dim]),
                                   other(x[self.dom.dim:])])
        return Function(newname, dom, cod, f)


class AxiomError(cat.AxiomError):
    """
    >>> f = Function('f', Dim(0), Dim(2), lambda x: x)
    >>> f.function([3, 2])
    [3, 2]
    >>> f([3, 2])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    function.AxiomError: Expected input of length 0, got 2 instead.
    """


class Id(Function):
    """
    Implements the identity function for a given dimension.

    >>> print(Id(5))
    Id(5)
    >>> assert Id(1)([476]) == np.array([476])
    >>> assert np.all(Id(2)([0, 1]) == np.array([0, 1]))
    """
    def __init__(self, dim):
        name = 'Id({})'.format(dim)
        dom = dim if isinstance(dim, Dim) else Dim(dim)
        super().__init__(name, dom, dom, lambda x: x)


class NumpyFunctor(MonoidalFunctor):
    """
    Implements functors into the category of functions on lists

    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> g = Box('g', y, x)
    >>> @discofunc(Dim(1), Dim(2))
    ... def copy(x):
    ...     return np.concatenate([x, x])
    >>> @discofunc(Dim(2), Dim(1))
    ... def add(x):
    ...     return np.array([x[0] + x[1]])
    >>> F = NumpyFunctor({x: Dim(1), y: Dim(2)}, {f: copy, g: add})
    >>> assert F(f >> g)([1]) == np.array([2])
    """
    def __call__(self, diagram):
        if isinstance(diagram, Ty):
            return sum([self.ob[Ty(x)] for x in diagram], Dim(0))
        if isinstance(diagram, Box):
            return super().__call__(diagram)
        if isinstance(diagram, Diagram):
            scan, result = diagram.dom, Id(self(diagram.dom))
            for box, off in zip(diagram.boxes, diagram.offsets):
                id_l = Id(self(scan[:off]))
                id_r = Id(self(scan[off + len(box.dom):]))
                result = result >> id_l @ self(box) @ id_r
                scan = scan[:off] + box.cod + scan[off + len(box.dom):]
            return result
        raise ValueError("Diagram expected, got {} of type {} "
                         "instead.".format(repr(diagram), type(diagram)))


def discofunc(dom, cod, name=False):
    """
    Decorator turning a python function into a discopy Function
    given domain and codomain information.

    >>> @discofunc(2, 2)
    ... def f(x):
    ...     return x[::-1]
    >>> assert isinstance(f, Function)
    >>> print(f)
    f
    >>> @discofunc(Dim(2), Dim(2), name='swap')
    ... def f(x):
    ...     return x[::-1]
    >>> print(f)
    swap
    """
    if isinstance(dom, int):
        dom = Dim(dom)
    if isinstance(cod, int):
        cod = Dim(cod)

    def decorator(f):
        if not name:
            return Function(f.__name__, dom, cod, f)
        else:
            return Function(name, dom, cod, f)
    return decorator
