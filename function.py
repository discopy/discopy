# -*- coding: utf-8 -*-
"""
Implements functors into the category of functions on vectors
with cartesian product as tensor.
"""

from discopy import messages
from discopy.cat import AxiomError
from discopy.moncat import Box, MonoidalFunctor
from discopy.circuit import PRO
from discopy.matrix import np


class Function(Box):
    """
    Wraps python functions with domain and codomain information.

    Parameters
    ----------
    name: str
        Name of the function.
    dom : int
        Domain of the diagram.
    cod : int
        Codomain of the diagram.
    function: any
        Python function with a call method.

    Notes
    -----

    When calling a 'Function' on a list, it is automatically turned into
    a Numpy array. It is sufficient that the input has a length which agrees
    with the domain dimension.

    >>> swap = Function('swap', 2, 2, lambda x: x[::-1])
    >>> assert np.all(swap([1, 2]) == swap(np.array([1, 2])))
    """
    def __init__(self, name, dom, cod, function):
        if isinstance(dom, PRO):
            dom = len(dom)
        if isinstance(cod, PRO):
            cod = len(cod)
        if not isinstance(dom, int):
            raise TypeError(messages.type_err(int, dom))
        if not isinstance(cod, int):
            raise TypeError(messages.type_err(int, cod))
        self._function = function
        if not isinstance(name, str):
            raise TypeError(messages.type_err(str, name))
        self._name = name
        super().__init__(name, PRO(dom), PRO(cod))

    @property
    def function(self):
        """
        The function stored in a discopy.Function object is immutable

        >>> f = Function('Id', PRO(2), PRO(2), lambda x: x)
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

        >>> f = Function('f', PRO(2), PRO(2), lambda x: x)
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
            Input list with 'len(value) == len(self.dom)'

        Notes
        -----
        When calling a 'Function' on a 'list', it is automatically turned into
        a Numpy/Jax array.

        >>> swap = Function('swap', 2, 2, lambda x: x[::-1])
        >>> assert np.all(swap([1, 2]) == swap(np.array([1, 2])))
        """
        if isinstance(value, list):
            value = np.array(value)
        if not len(value) == len(self.dom):
            raise AxiomError("Expected input of length {}, got {} instead."
                             .format(len(self.dom), len(value)))
        return self.function(value)

    def then(self, other):
        """
        Returns the sequential composition of 'self' with 'other'.
        This method is called using the binary operators `>>` and `<<`.

        >>> swap = Function('swap', PRO(2), PRO(2), lambda x: x[::-1])
        >>> abs = Function('abs', PRO(2), PRO(2), np.absolute)
        >>> assert np.all((swap >> abs)([14, 42]) == (abs << swap)([14, 42]))
        """
        if not isinstance(other, Function):
            raise TypeError(messages.type_err(Function, other))
        if len(self.cod) != len(other.dom):
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        if not isinstance(self, Id):
            if not isinstance(other, Id):
                newname = '(' + self.name + ' >> ' + other.name + ')'
            else:
                newname = self.name
        else:
            newname = other.name

        def func(val):
            return other(self(val))
        return Function(newname, self.dom, other.cod, func)

    def tensor(self, other):
        """
        Returns the parallel composition of 'self' and 'other'.
        This method is called using the binary operator `@`.

        >>> add = Function('add', PRO(2), PRO(1),
        ...                lambda x: np.array([x[0] + x[1]]))
        >>> copy = Function('copy', PRO(1), PRO(2),\\
        ...                 lambda x: np.concatenate([x, x]))
        >>> assert np.all((add @ copy)([3, 1, 2]) == np.array([4, 2, 2]))
        """
        if not isinstance(other, Function):
            raise TypeError(messages.type_err(Function, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        if not self.name == 'Id(0)':
            if not other.name == 'Id(0)':
                newname = '(' + self.name + ' @ ' + other.name + ')'
            else:
                newname = self.name
        else:
            newname = other.name

        def func(val):
            return np.concatenate([self(val[:len(self.dom)]),
                                   other(val[len(self.dom):])])
        return Function(newname, dom, cod, func)

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
        dom = dim if isinstance(dim, int) else len(dim)
        super().__init__(name, dom, dom, lambda x: x)


class CartesianFunctor(MonoidalFunctor):
    """
    Implements functors into the category of functions on lists

    >>> from discopy import Ty
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> g = Box('g', y, x)
    >>> @discofunc(PRO(1), PRO(2))
    ... def copy(x):
    ...     return np.concatenate([x, x])
    >>> @discofunc(PRO(2), PRO(1))
    ... def add(x):
    ...     return np.array([x[0] + x[1]])
    >>> F = CartesianFunctor({x: PRO(1), y: PRO(2)}, {f: copy, g: add})
    >>> assert F(f >> g)([1]) == np.array([2])
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_cls=PRO, ar_cls=Function)


def discofunc(dom, cod, name=None):
    """
    Decorator turning a python function into a discopy Function
    given domain and codomain information.

    >>> @discofunc(2, 2)
    ... def f(x):
    ...     return x[::-1]
    >>> assert isinstance(f, Function)
    >>> print(f)
    f
    >>> @discofunc(PRO(2), PRO(2), name='swap')
    ... def f(x):
    ...     return x[::-1]
    >>> print(f)
    swap
    """
    if isinstance(dom, int):
        dom = PRO(dom)
    if isinstance(cod, int):
        cod = PRO(cod)

    def decorator(func):
        if name is None:
            return Function(func.__name__, dom, cod, func)
        return Function(name, dom, cod, func)
    return decorator
