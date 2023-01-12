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
        get_backend

See also
--------

* :class:`Tensor` is a subclass of :class:`Matrix` with the Kronecker product
  as tensor.
* :class:`Matrix` is used to evaluate :class:`quantum.optics.Diagram`.

"""
from __future__ import annotations

from contextlib import contextmanager

from discopy import cat, monoidal, config
from discopy.cat import (
    factory,
    Composable,
    assert_iscomposable,
    assert_isparallel,
)
from discopy.monoidal import Whiskerable
from discopy.utils import assert_isinstance


@factory
class Matrix(Composable[int], Whiskerable):
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

    The default data type is ``int``, but this can be changed if necessary.

    >>> Matrix.dtype = float
    >>> assert Matrix == Matrix[float] != Matrix[int]
    >>> Matrix.dtype = int
    >>> assert Matrix == Matrix[int] != Matrix[float]

    The data type needs to have the structure of a rig (riNg with no negatives)
    i.e. with methods ``__add__`` and ``__mul__`` as well as an ``__init__``
    that can accept both ``0`` and ``1`` as input.

    Examples
    --------
    >>> m = Matrix([0, 1, 1, 0], 2, 2)
    >>> v = Matrix([0, 1], 1, 2)
    >>> v >> m >> v.dagger()
    Matrix([0], dom=1, cod=1)
    >>> m + m
    Matrix([0, 2, 2, 0], dom=2, cod=2)
    >>> assert m.then(m, m, m, m) == m >> m >> m >> m >> m

    The monoidal product for :py:class:`.Matrix` is the direct sum:

    >>> x = Matrix([2, 4], 2, 1)
    >>> x.array
    array([[2],
           [4]])
    >>> x @ x
    Matrix([2, 0, 4, 0, 0, 2, 0, 4], dom=4, cod=2)
    >>> (x @ x).array
    array([[2, 0],
           [4, 0],
           [0, 2],
           [0, 4]])
    """
    dtype = int

    def __class_getitem__(cls, dtype: type, _cache=dict()):
        if cls.dtype not in _cache or _cache[cls.dtype] != cls:
            _cache.clear()
            _cache[cls.dtype] = cls  # Ensure Matrix == Matrix[Matrix.dtype].
        if dtype not in _cache:
            class C(cls.factory):
                pass

            C.__name__ = C.__qualname__ = \
                f"{cls.factory.__name__}[{dtype.__name__}]"
            C.dtype = dtype
            _cache[dtype] = C
        return _cache[dtype]

    def cast_dtype(self, dtype: type) -> Matrix:
        """
        Cast a matrix to a given ``dtype``.

        Parameters:
            dtype : The target datatype.

        Example
        -------
        >>> assert Matrix.id().cast_dtype(bool) == Matrix[bool].id()
        """
        return type(self)[dtype](self.array, self.dom, self.cod)

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

    def is_close(self, other: Matrix) -> bool:
        """
        Whether a matrix is numerically close to an ``other``.

        Parameters:
            other : The other matrix with which to check closeness.
        """
        assert_isinstance(other, type(self))
        assert_isinstance(self, type(other))
        assert_isparallel(self, other)
        with backend() as np:
            return np.isclose(self.array, other.array).all()

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
            return cls(np.identity(dom, cls.dtype), dom, dom)

    def then(self, other: Matrix = None, *others: Matrix) -> Matrix:
        if others or other is None:
            return cat.Arrow.then(self, other, *others)
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
        array = Matrix.zero(dom, cod).array
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
            return cls(np.zeros((dom, cod)), dom, cod)

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
        Matrix([0, 1, 1, 0], dom=2, cod=2)
        """
        dom = cod = left + right
        array = Matrix.zero(dom, cod).array
        array[right:, :left] = Matrix.id(left).array
        array[:right, left:] = Matrix.id(right).array
        return cls(array, dom, cod)

    def transpose(self) -> Matrix:
        return type(self)(self.array.transpose(), self.cod, self.dom)

    def conjugate(self) -> Matrix:
        return type(self)(self.array.conjugate(), self.dom, self.cod)

    def dagger(self) -> Matrix:
        return self.conjugate().transpose()

    def map(self, func: Callable[[dtype], dtype], dtype=None) -> Matrix:
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
        Matrix([0, 0, 1, 0], dom=1, cod=4)
        """
        return cls([[i == j for j in range(x)]], x ** 0, x)

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

    def trace(self, n=1) -> Matrix:
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
    def __init__(self, module, array=None):
        self.module, self.array = module, array or module.array

    def __getattr__(self, attr):
        return getattr(self.module, attr)


class NumPy(Backend):
    def __init__(self):
        import numpy
        super().__init__(numpy)


class JAX(Backend):
    def __init__(self):
        import jax
        super().__init__(jax.numpy)


class PyTorch(Backend):
    def __init__(self):
        import torch
        super().__init__(torch, array=torch.as_tensor)


class TensorFlow(Backend):
    def __init__(self):
        import tensorflow.experimental.numpy as tnp
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()
        super().__init__(tnp)


BACKENDS = {
    'np': NumPy,
    'numpy': NumPy,
    'jax': JAX,
    'jax.numpy': JAX,
    'pytorch': PyTorch,
    'torch': PyTorch,
    'tensorflow': TensorFlow,
}


@contextmanager
def backend(name=None, _stack=[config.DEFAULT_BACKEND], _cache=dict()):
    name = name or _stack[-1]
    _stack.append(name)
    try:
        if name not in _cache:
            _cache[name] = BACKENDS[name]()
        yield _cache[name]
    finally:
        _stack.pop()


def get_backend():
    with backend() as result:
        return result
