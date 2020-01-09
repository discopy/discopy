# -*- coding: utf-8 -*-
"""
Implements the symmetric monoidal category (PROP) of functions on Numpy lists
with cartesian product as tensor.

Projections and the copy map witness the categorical product.

>>> proj0 = Function('proj0', Dim(2), Dim(1), lambda x: np.array(x[0]))
>>> proj1 = Function('proj1', Dim(2), Dim(1), lambda x: np.array(x[1]))
>>> @discofunc(1, 2)
... def copy(x):
...     return np.concatenate([x, x])
>>> assert (copy >> proj0)([46]) == Id(1)([46]) == (copy >> proj1)([46])

'Dim(0)' is a terminal object with the following discarding map.

>>> discard = lambda n: Function('discard', Dim(n), Dim(0), lambda x: [])
>>> assert discard(3)([23, 2, 67]) == [] == discard(1)([23])

We can check the axioms for symmetry on specific inputs.

>>> swap = Function('swap', Dim(2), Dim(2), lambda x: x[::-1])
>>> assert np.all((swap >> swap)([1, 2]) == Id(2)([1, 2]))
>>> assert np.all((Id(1) @ swap >> swap @ Id(1) >> Id(1) @ swap)([0, 1, 2])
...            == (swap @ Id(1) >> Id(1) @ swap >> swap @ Id(1))([0, 1, 2]))

Notes
-----

The name of a composition is the composition of the names
with brackets.

>>> (swap >> swap >> abs).name
'((swap >> swap) >> abs)'
>>> print(swap >> abs)
(swap >> abs)
>>> assert (swap >> Id(2)).name == 'swap'

The name of a tensor is the tensor of the names
with brackets.

>>> assert (add @ copy @ Id(0)).name == '(add @ copy)'
>>> print(Id(0) @ Id(0))
Id(0)
"""
from functools import reduce as fold
from discopy import cat
from cat import AxiomError
from discopy.matrix import np
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor


class Dim(Ty):
    """ Implements dimensions as non-negative integers.
    These form a monoid with sum as product denoted by '@' and unit 'Dim(0)'.

    Parameters
    ----------
    x : int
        Non-negative integer value of Dim.

    >>> assert Dim(0) @ Dim(4) == Dim(4) @ Dim(0) == Dim(4)
    >>> assert Dim(2) @ Dim(3) == Dim(5)
    >>> assert sum([Dim(0), Dim(3), Dim(4)], Dim(0)) == Dim(7)
    """
    def __init__(self, x=0):
        if not isinstance(x, int) or x < 0:
            raise ValueError("Expected non-negative integer, "
                             "got {} instead.".format(repr(x)))
        self._dim = x
        super().__init__(*[Ob(1) for i in range(x)])

    @property
    def dim(self):
        """
        Returns the integer value stored in a Dim object.

        >>> assert Dim(5).dim == 5
        """
        return self._dim

    def __matmul__(self, other):
        return Dim(self.dim + other.dim)

    def __add__(self, other):
        return self @ other

    def __repr__(self):
        return "Dim({})".format(self.dim)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(repr(self))


class Function(Box):
    """
    Wraps python functions with domain and codomain information.

    Parameters
    ----------
    name: 'string'
        Name of the function.
    dom : 'function.Dim' or non-negative 'int'
        Domain of the diagram.
    cod : 'function.Dim' or non-negative 'int'
        Codomain of the diagram.
    function: any
        Python function with a call method.

    Notes
    -----
    When calling a 'Function' on a list, it is automatically turned into
    a Numpy array.

    >>> swap = Function('swap', 2, 2, lambda x: x[::-1])
    >>> assert np.all(swap([1, 2]) == swap(np.array([1, 2])))

    As an example, we show that copy and add satisfy the bimonoid law.

    >>> copy = Function('copy', 1, 2, lambda x: np.concatenate([x, x]))
    >>> assert np.all(copy([1]) == np.array([1, 1]))
    >>> add = Function('add', 2, 1, lambda x: np.sum(x, keepdims=True))
    >>> assert np.all(add([1, 2]) == np.array([3]))
    >>> assert np.all((copy @ copy >> Id(1) @ swap @ Id(1)
    ...                >> add @ add)([123, 25]) == (add >> copy)([123, 25]))
    """
    def __init__(self, name, dom, cod, function):
        if isinstance(dom, int):
            dom = Dim(dom)
        if isinstance(cod, int):
            cod = Dim(cod)
        if not isinstance(dom, Dim):
            raise ValueError(
                "Dim expected for dom, got {} instead.".format(type(dom)))
        if not isinstance(cod, Dim):
            raise ValueError(
                "Dim expected for cod, got {} instead.".format(type(cod)))
        self._function = function
        if not isinstance(name, str):
            raise ValueError(
                "String expected for name, got {} instead.".format(type(name)))
        self._name = name
        super().__init__(name, dom, cod)

    @property
    def function(self):
        """
        The function stored in a discopy.Function object is immutable

        >>> f = Function('Id', Dim(2), Dim(2), lambda x: x)
        >>> assert f.function(1) == 1
        >>> f.function = lambda x: 2*x  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
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
        return "Function(name={}, dom={}, cod={}, function={})".format(
            self.name, self.dom, self.cod, repr(self.function))

    def __str__(self):
        return self.name

    def __call__(self, value):
        """
        In order to call a Function, it is sufficient that the input
        has a length which agrees with the domain dimension.

        Parameters
        ----------
        value : 'list' or 'numpy.ndarray' or 'jax.interpreters.xla.DeviceArray'
            Input list with 'len(value) == self.dom.dim'


        Notes
        -----
        When calling a 'Function' on a 'list', it is automatically turned into
        a Numpy/Jax array.

        >>> swap = Function('swap', 2, 2, lambda x: x[::-1])
        >>> assert np.all(swap([1, 2]) == swap(np.array([1, 2])))
        """
        if isinstance(value, list):
            value = np.array(value)
        if not len(value) == self.dom.dim:
            raise AxiomError("Expected input of length {}, got {} instead."
                             .format(self.dom.dim, len(value)))
        return self.function(value)

    def then(self, other):
        """
        Returns the sequential composition of 'self' with 'other'.
        This method is called using the binary operators `>>` and `<<`.

        >>> swap = Function('swap', Dim(2), Dim(2), lambda x: x[::-1])
        >>> abs = Function('abs', Dim(2), Dim(2), np.absolute)
        >>> assert np.all((swap >> abs)([14, 42]) == (abs << swap)([14, 42]))
        """
        if not isinstance(other, Function):
            raise ValueError(
                "Function expected, got {} instead.".format(repr(other)))
        if self.cod.dim != other.dom.dim:
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        if not isinstance(self, Id):
            if not isinstance(other, Id):
                newname = '(' + self.name + ' >> ' + other.name + ')'
            else:
                newname = self.name
        else:
            newname = other.name

        def f(x):
            return other(self(x))
        return Function(newname, self.dom, other.cod, f)

    def tensor(self, other):
        """
        Returns the parallel composition of 'self' and 'other'.
        This method is called using the binary operator `@`.

        >>> add = Function('add', Dim(2), Dim(1),
        ...                lambda x: np.array([x[0] + x[1]]))
        >>> copy = Function('copy', Dim(1), Dim(2),\\
        ...                 lambda x: np.concatenate([x, x]))
        >>> assert np.all((add @ copy)([3, 1, 2]) == np.array([4, 2, 2]))
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

    @staticmethod
    def id(x):
        """
        >>> assert np.all(Function.id(2)([1, 2]) == np.array([1, 2]))
        """
        return Id(x)


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


class Copy(Function):
    """
    Implements the copy function with domain 'dom' and codomain 'dom * copies'.
    """
    def __init__(self, dom, copies=2):
        name = 'Copy({}, {})'.format(dom, copies)

        def function(x):
            return np.concatenate([x for x in range(copies)])
        super().__init__(name, dom, copies * dom, function)


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
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_cls=Dim, ar_cls=Function)


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
