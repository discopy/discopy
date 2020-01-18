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
    Wraps a python function with domain and codomain information.

    >>> SWAP = Function(2, 2, lambda x: x[::-1])
    >>> COPY = Function(1, 2, lambda x: np.concatenate((x, x)))
    >>> ADD = Function(2, 1, lambda x: np.sum(x, keepdims=True))

    Parameters
    ----------
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

    >>> assert np.all(SWAP([1, 2]) == SWAP(np.array([1, 2])))
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
        super().__init__(None, PRO(dom), PRO(cod))

    @property
    def function(self):
        """
        The function stored in a discopy.Function object is immutable

        >>> Id = Function(PRO(2), PRO(2), lambda x: x)
        >>> assert Id.function(1) == 1
        >>> Id.function = lambda x: 2*x  # doctest: +ELLIPSIS
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

        >>> assert np.all(SWAP([1, 2]) == SWAP(np.array([1, 2])))
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

        >>> abs = Function(PRO(2), PRO(2), np.absolute)
        >>> assert np.all((SWAP >> abs)([14, 42]) == (abs << SWAP)([14, 42]))
        """
        if not isinstance(other, Function):
            raise TypeError(messages.type_err(Function, other))
        if len(self.cod) != len(other.dom):
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))

        def func(val):
            return other(self(val))
        return Function(self.dom, other.cod, func)

    def tensor(self, other):
        """
        Returns the parallel composition of 'self' and 'other'.
        This method is called using the binary operator `@`.

        >>> assert np.all((ADD @ COPY)([3, 1, 2]) == np.array([4, 2, 2]))
        """
        if not isinstance(other, Function):
            raise TypeError(messages.type_err(Function, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod

        def func(val):
            return np.concatenate([self(val[:len(self.dom)]),
                                   other(val[len(self.dom):])])
        return Function(dom, cod, func)

    @staticmethod
    def id(x):
        """
        >>> assert np.all(Function.id(2)([1, 2]) == np.array([1, 2]))
        """
        return Id(x)


class Id(Function):
    """
    Implements the identity function for a given dimension.

    >>> assert Id(1)([476]) == np.array([476])
    >>> assert np.all(Id(2)([0, 1]) == np.array([0, 1]))
    """
    def __init__(self, dim):
        dom = dim if isinstance(dim, int) else len(dim)
        super().__init__(dom, dom, lambda x: x)


class CartesianFunctor(MonoidalFunctor):
    """
    Implements functors into the PRO of functions.

    >>> from discopy import Ty
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> g = Box('g', y, x)
    >>> F = CartesianFunctor({x: PRO(1), y: PRO(2)}, {f: COPY, g: ADD})
    >>> assert F(f >> g)([1]) == np.array([2])
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_cls=PRO, ar_cls=Function)


SWAP = Function(2, 2, lambda x: x[::-1])
COPY = Function(1, 2, lambda x: np.concatenate((x, x)))
ADD = Function(2, 1, lambda x: np.sum(x, keepdims=True))
