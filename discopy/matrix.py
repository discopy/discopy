"""
The category of matrices with the direct sum as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Matrix
    Backend
    NumPy
    JAX
    PyTorch
    TensorFlow

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        backend
        set_backend
        get_backend

See also
--------

* :class:`discopy.tensor.Tensor` is a subclass of :class:`Matrix` with the
  Kronecker product as tensor.

"""
from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Union, Literal as L, Callable, TYPE_CHECKING

from discopy import monoidal, config, messages
from discopy.cat import (
    factory,
    Composable,
    assert_iscomposable,
    assert_isparallel,
)
from discopy.monoidal import Whiskerable
from discopy.utils import assert_isinstance, unbiased, NamedGeneric

if TYPE_CHECKING:
    import sympy


@factory
class Matrix(Composable[int], Whiskerable, NamedGeneric['dtype']):
    """
    A matrix is an ``array`` with natural numbers as ``dom`` and ``cod``.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            zero
            swap
            transpose
            conjugate
            dagger
            map
            round
            copy
            discard
            merge
            basis
            repeat
            trace
            lambdify
            subs
            grad

    Note
    ----
    The class ``Matrix[dtype]`` has arrays with entries in any given ``dtype``.
    For example:

    >>> Matrix[complex].id(1)
    Matrix[complex]([1.+0.j], dom=1, cod=1)
    >>> assert Matrix[complex].id(1) != Matrix[float].id(1)

    The default data type is determined by underlying array datastructure of
    the backend used. An array is initialised with ``array`` as parameter and
    the dtype of the ``Matrix`` object is the data type of this array.
    >>> import numpy as np
    >>> assert Matrix([1, 0], dom=1, cod=2).dtype == np.int64
    >>> assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float64
    >>> assert Matrix([0.5j], dom=1, cod=1).dtype == np.complex128

    The data type needs to have the structure of a rig (riNg with no negatives)
    i.e. with methods ``__add__`` and ``__mul__`` as well as an ``__init__``
    that can accept both ``0`` and ``1`` as input.

    Examples
    --------
    >>> m = Matrix([0, 1, 1, 0], 2, 2)
    >>> v = Matrix([0, 1], 1, 2)
    >>> v >> m >> v.dagger()
    Matrix[int64]([0], dom=1, cod=1)
    >>> m + m
    Matrix[int64]([0, 2, 2, 0], dom=2, cod=2)
    >>> assert m.then(m, m, m, m) == m >> m >> m >> m >> m

    The monoidal product for :py:class:`.Matrix` is the direct sum:

    >>> x = Matrix([2, 4], 2, 1)
    >>> x.array
    array([[2],
           [4]])
    >>> x @ x
    Matrix[int64]([2, 0, 4, 0, 0, 2, 0, 4], dom=4, cod=2)
    >>> (x @ x).array
    array([[2, 0],
           [4, 0],
           [0, 2],
           [0, 4]])
    """

    def cast(self, dtype: type) -> Matrix:
        """
        Cast a matrix to a given ``dtype``.

        Parameters:
            dtype : The target datatype.

        Example
        -------
        >>> assert Matrix.id().cast(bool) == Matrix[bool].id()
        """
        return type(self)[dtype](self.array, self.dom, self.cod)

    def __new__(cls, array, *args, **kwargs):
        with backend() as np:
            if cls.dtype is None:
                array = np.array(array)
                # The dtype of an np.arrays is a class that contains a type
                # attribute that is the actual type. However, other backends
                # have different structures, so this is the easiest option:
                dtype = getattr(array.dtype, "type", array.dtype)
                return cls.__new__(cls[dtype], array, *args, **kwargs)
            return object.__new__(cls)

    def __init__(self, array, dom: int, cod: int):
        assert_isinstance(dom, int)
        assert_isinstance(cod, int)
        self.dom, self.cod = dom, cod
        with backend() as np:
            self.array = np.array(array, dtype=self.dtype).reshape((dom, cod))

    def __eq__(self, other):
        return isinstance(other, self.factory)\
            and self.dtype == other.dtype\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and (self.array == other.array).all()

    def is_close(self, other: Matrix, rtol: float = 1.e-8, atol: float = 1.e-8
                 ) -> bool:
        """
        Whether a matrix is numerically close to an ``other``.

        Parameters:
            other : The other matrix with which to check closeness.
            rtol:
                The relative tolerance parameter (see Notes).
                Default value for results of order unity is 1.e-5
            atol :
                The absolute tolerance parameter (see Notes).
                Default value for results of order unity is 1.e-8

        Notes:
        (taken from np.isclose documentation)

            For finite values, isclose uses the following equation to
            test whether two floating point values are equivalent.

            absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

            Unlike the built-in `math.isclose`, the above equation is not
            symmetric in `a` and `b` -- it assumes `b` is the reference
            value -- so that `isclose(a, b)` might be different from
            `isclose(b, a)`.

            Furthermore, the default value of atol is not zero, and is used
            to determine what small values should be considered close to zero.
            The default value is appropriate for expected values of order
            unity: if the expected values are significantly smaller than one,
            it can result in false positives.

            `atol` should be carefully selected for the use case at hand.
            A zero value for `atol` will result in `False` if either `a`
            or `b` is zero.

            `isclose` is not defined for non-numeric data types.
            `bool` is considered a numeric data-type for this purpose

        """
        assert_isinstance(other, type(self))
        assert_isinstance(self, type(other))
        assert_isparallel(self, other)
        with backend() as np:
            return np.isclose(self.array, other.array, rtol, atol).all()

    def __repr__(self):
        np_array = getattr(self.array, 'numpy', lambda: self.array)()
        return type(self).__name__ + f"({array2string(np_array.reshape(-1))},"\
                                     f" dom={self.dom}, cod={self.cod})"

    def __iter__(self):
        for i in self.array:
            yield i

    def __bool__(self):
        return bool(self.array)

    def __int__(self):
        return int(self.array)

    def __float__(self):
        return float(self.array)

    def __complex__(self):
        return complex(self.array)

    @classmethod
    def id(cls, dom=0) -> Matrix:
        with backend('numpy') as np:
            return cls(np.identity(dom, dtype=cls.dtype or int), dom, dom)

    twist = id

    @unbiased
    def then(self, other: Matrix) -> Matrix:
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        with backend() as np:
            array = np.matmul(self.array, other.array)
        return type(self)(array, self.dom, other.cod)

    def tensor(self, other: Matrix = None, *others: Matrix):
        if others or other is None:
            return monoidal.Diagram.tensor(self, other, *others)
        assert_isinstance(other, type(self))
        dom, cod = self.dom + other.dom, self.cod + other.cod
        array = self.zero(dom, cod).array
        array[:self.dom, :self.cod] = self.array
        array[self.dom:, self.cod:] = other.array
        return type(self)(array, dom, cod)

    def __add__(self, other):
        assert_isinstance(other, Matrix)
        assert_isparallel(self, other)
        return type(self)(self.array + other.array, self.dom, self.cod)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    @classmethod
    def zero(cls, dom: int, cod: int) -> Matrix:
        """
        Returns the zero matrix of a given shape.

        Examples
        --------
        >>> assert Matrix.zero(2, 2) == Matrix([0, 0, 0, 0], 2, 2)
        """
        with backend() as np:
            return cls(np.zeros((dom, cod), dtype=cls.dtype or int), dom, cod)

    @classmethod
    def swap(cls, left: int, right: int) -> Matrix:
        """
        The matrix that swaps left and right dimensions.

        Parameters:
            left : The left dimension.
            right : The right dimension.

        Example
        -------
        >>> Matrix.swap(1, 1)
        Matrix[int64]([0, 1, 1, 0], dom=2, cod=2)
        >>> Matrix.swap(2,1)
        Matrix[int64]([0, 1, 0, 0, 0, 1, 1, 0, 0], dom=3, cod=3)
        """
        dom = cod = left + right
        array = Matrix.zero(dom, cod).array
        array[:left, right:] = Matrix.id(left).array
        array[left:, :right] = Matrix.id(right).array
        return cls(array, dom, cod)

    braid = swap

    def transpose(self) -> Matrix:
        return type(self)(self.array.transpose(), self.cod, self.dom)

    def conjugate(self) -> Matrix:
        return type(self)(self.array.conjugate(), self.dom, self.cod)

    def dagger(self) -> Matrix:
        return self.conjugate().transpose()

    def map(self, func: Callable, dtype: type | None = None) -> Matrix:
        array = list(map(func, self.array.reshape(-1)))
        return type(self)[dtype or self.dtype](array, self.dom, self.cod)

    def round(self, decimals=0) -> Matrix:
        """ Rounds the entries of a matrix up to a number of decimals. """
        with backend() as np:
            array = np.around(self.array, decimals=decimals)
        return type(self)(array, self.dom, self.cod)

    @classmethod
    def copy(cls, x: int, n: int) -> Matrix:
        array = [[i + int(j % n * x) == j
                  for j in range(n * x)] for i in range(x)]
        return cls(array, x, n * x)

    @classmethod
    def discard(cls, x: int) -> Matrix:
        return cls.copy(x, 0)

    @classmethod
    def merge(cls, x: int, n: int) -> Matrix:
        return cls.copy(x, n).dagger()

    @classmethod
    def ones(cls, x: int) -> Matrix:
        return cls.merge(x, 0)

    @classmethod
    def basis(cls, x: int, i: int) -> Matrix:
        """
        The ``i``-th basis vector of dimension ``x``.

        Parameters:
            x : The dimension of the basis vector.
            i : The index of the basis vector.

        Example
        -------
        >>> Matrix.basis(4, 2)
        Matrix[int64]([0, 0, 1, 0], dom=1, cod=4)
        """
        return cls([[int(i == j) for j in range(x)]], x ** 0, x)

    def repeat(self) -> Matrix:
        """
        The reflexive transitive closure of a boolean matrix.

        Example
        -------
        >>> Matrix[bool]([0, 1, 1, 0], 2, 2).repeat()
        Matrix[bool]([True, True, True, True], dom=2, cod=2)
        """
        if self.dtype != bool or self.dom != self.cod:
            raise TypeError(messages.MATRIX_REPEAT_ERROR)
        return sum(
            self.id(self.dom).then(*n * [self]) for n in range(self.dom + 1))

    def trace(self, n=1, left=False) -> Matrix:
        """
        The trace of a Boolean matrix, computed with :meth:`Matrix.repeat`.

        Parameters:
            n : The number of dimensions to trace.

        Example
        -------
        >>> assert Matrix[bool].swap(1, 1).trace() == Matrix[bool].id(1)
        """
        A, B, C, D = (row >> self >> column
                      for row in [self.id(self.dom - n) @ self.ones(n),
                                  self.ones(self.dom - n) @ self.id(n)]
                      for column in [self.id(self.cod - n) @ self.discard(n),
                                     self.discard(self.cod - n) @ self.id(n)])
        return A + (B >> D.repeat() >> C)

    def lambdify(
            self, *symbols: "sympy.Symbol", dtype=None, **kwargs) -> Callable:
        from sympy import lambdify
        with backend() as np:
            array = lambdify(symbols, self.array, modules=np.module, **kwargs)
        dtype = dtype or self.dtype
        return lambda *xs: type(self)[dtype](array(*xs), self.dom, self.cod)

    def subs(self, *args) -> Matrix:
        return self.map(lambda x: getattr(x, "subs", lambda y, *_: y)(*args))

    def grad(self, var, **params) -> Matrix:
        """ Gradient with respect to variables. """
        return self.map(lambda x:
                        getattr(x, "diff", lambda _: 0)(var, **params))


def array2string(array, **params):
    """ Numpy array pretty print. """
    import numpy
    numpy.set_printoptions(threshold=config.NUMPY_THRESHOLD)
    return numpy.array2string(array, **dict(params, separator=', '))\
        .replace('[ ', '[').replace('  ', ' ')


class Backend:
    """
    A matrix backend.

    Parameters:
        module : The main module of the backend.
        array : The array class of the backend.
    """
    def __init__(self, module: ModuleType, array: type = None):
        self.module, self.array = module, array or module.array

    def __getattr__(self, attr):
        return getattr(self.module, attr)


class NumPy(Backend):
    """ NumPy backend. """
    def __init__(self):
        import numpy
        super().__init__(numpy)


class JAX(Backend):
    """ JAX backend. """
    def __init__(self):
        import jax
        super().__init__(jax.numpy)


class PyTorch(Backend):
    """ PyTorch backend. """
    def __init__(self):
        import torch
        super().__init__(torch, array=torch.as_tensor)


class TensorFlow(Backend):
    """ TensorFlow backend. """
    def __init__(self):
        import tensorflow.experimental.numpy as tnp
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()
        super().__init__(tnp)


BACKENDS = {
    'numpy': NumPy,
    'jax': JAX,
    'pytorch': PyTorch,
    'tensorflow': TensorFlow,
}

BackendName = Union[tuple(L[x] for x in BACKENDS)]


@contextmanager
def backend(name: BackendName = None,
            _stack=[config.DEFAULT_BACKEND], _cache=dict()):
    """
    Context manager for matrix backend.

    Parameters:
        name : The name of the backend, default is ``"numpy"``.

    Example
    -------
    >>> with backend('jax'):
    ...     assert type(Matrix([0, 1, 1, 0], 2, 2).array).__module__\\
    ...         == 'jaxlib.xla_extension'
    """
    name = name or _stack[-1]
    _stack.append(name)
    try:
        if name not in _cache:
            _cache[name] = BACKENDS[name]()
        yield _cache[name]
    finally:
        _stack.pop()


def set_backend(name: BackendName) -> None:
    """
    Override the default backend.

    Parameters:
        name : The name of the backend.

    Example
    -------
    >>> set_backend('jax')
    >>> assert type(Matrix([0, 1, 1, 0], 2, 2).array).__module__\\
    ...     == 'jaxlib.xla_extension'
    >>> set_backend('numpy')
    >>> assert type(Matrix([0, 1, 1, 0], 2, 2).array).__module__\\
    ...     == 'numpy'
    """
    backend.__wrapped__.__defaults__[1][-1] = name


def get_backend() -> Backend:
    """
    Get the current backend.

    Example
    -------
    >>> set_backend('jax')
    >>> assert isinstance(get_backend(), JAX)
    >>> set_backend('numpy')
    >>> assert isinstance(get_backend(), NumPy)
    """
    with backend() as result:
        return result
