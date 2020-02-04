# -*- coding: utf-8 -*-

"""
Implements dagger monoidal functors into matrices.

>>> n = Ty('n')
>>> Alice, Bob = Box('Alice', Ty(), n), Box('Bob', Ty(), n)
>>> loves = Box('loves', n, n)
>>> ob, ar = {n: 2}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = MatrixFunctor(ob, ar)
>>> F(Alice >> loves >> Bob.dagger())
Matrix(dom=Dim(1), cod=Dim(1), array=[1])
"""

import functools

from discopy import messages
from discopy.cat import AxiomError
from discopy.rigidcat import Ob, Ty, Box, Cup, Cap, Diagram, Functor

try:
    import warnings
    for msg in messages.IGNORE_WARNINGS:
        warnings.filterwarnings("ignore", message=msg)
    import jax.numpy as np
except ImportError:  # pragma: no cover
    import numpy as np


class Dim(Ty):
    """ Implements dimensions as tuples of positive integers.
    Dimensions form a monoid with product @ and unit Dim(1).

    >>> Dim(1) @ Dim(2) @ Dim(3)
    Dim(2, 3)
    """
    def __init__(self, *dims):
        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError(messages.type_err(int, dim))
            if dim < 1:
                raise ValueError
        super().__init__(*[Ob(dim) for dim in dims if dim > 1])

    def tensor(self, other):
        return Dim(*[x.name for x in super().tensor(other)])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Dim(*[x.name for x in super().__getitem__(key)])
        return super().__getitem__(key).name

    def __repr__(self):
        return "Dim({})".format(', '.join(map(repr, self)) or '1')

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(repr(self))

    @property
    def l(self):
        """
        >>> assert Dim(2, 3, 4).l == Dim(4, 3, 2)
        """
        return Dim(*self[::-1])

    @property
    def r(self):
        """
        >>> assert Dim(2, 3, 4).r == Dim(4, 3, 2)
        """
        return Dim(*self[::-1])


class Matrix(Box):
    """ Implements a matrix with dom, cod and numpy array.

    >>> m = Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
    >>> v = Matrix(Dim(1), Dim(2), [0, 1])
    >>> v >> m >> v.dagger()
    Matrix(dom=Dim(1), cod=Dim(1), array=[0])
    """
    def __init__(self, dom, cod, array):
        self._array = np.array(array).reshape(dom + cod)
        super().__init__(array, dom, cod)

    @property
    def array(self):
        """ Numpy array. """
        return self._array

    def __bool__(self):
        return bool(self.array)

    def __repr__(self):
        return "Matrix(dom={}, cod={}, array={})".format(
            self.dom, self.cod, list(self.array.flatten()))

    def __str__(self):
        return repr(self)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(messages.type_err(Matrix, other))
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        return Matrix(self.dom, self.cod, self.array + other.array)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return self.array == other
        return (self.dom, self.cod) == (other.dom, other.cod)\
            and np.all(self.array == other.array)

    def then(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(messages.type_err(Matrix, other))
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        array = np.tensordot(self.array, other.array, len(self.cod))\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        return Matrix(self.dom, other.cod, array)

    def tensor(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(messages.type_err(Matrix, other))
        dom, cod = self.dom + other.dom, self.cod + other.cod
        array = np.tensordot(self.array, other.array, 0)\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        return Matrix(dom, cod, array)

    def dagger(self):
        array = np.moveaxis(
            self.array, range(len(self.dom + self.cod)),
            [i + len(self.cod) if i < len(self.dom) else
             i - len(self.dom) for i in range(len(self.dom + self.cod))])
        return Matrix(self.cod, self.dom, np.conjugate(array))

    @staticmethod
    def id(x):
        return Id(x)

    @staticmethod
    def cups(left, right):
        if not isinstance(left, Dim):
            raise TypeError(messages.type_err(Dim, left))
        if not isinstance(right, Dim):
            raise TypeError(messages.type_err(Dim, right))
        if left.r != right:
            raise AxiomError(messages.are_not_adjoints(left, right))
        return Matrix(left @ right, Dim(1), Id(left).array)

    @staticmethod
    def caps(left, right):
        return Matrix.cups(left, right).dagger()


class Id(Matrix):
    """ Implements the identity matrix for a given dimension.

    >>> Id(1)
    Matrix(dom=Dim(1), cod=Dim(1), array=[1])
    >>> Id(2)
    Matrix(dom=Dim(2), cod=Dim(2), array=[1.0, 0.0, 0.0, 1.0])
    >>> Id(1, 2, 3)  # doctest: +ELLIPSIS
    Matrix(dom=Dim(2, 3), cod=Dim(2, 3), array=[1.0, ..., 1.0])
    """
    def __init__(self, *dim):
        """
        >>> Id(1)
        Matrix(dom=Dim(1), cod=Dim(1), array=[1])
        >>> list(Id(2).array.flatten())
        [1.0, 0.0, 0.0, 1.0]
        >>> Id(2).array.shape
        (2, 2)
        >>> list(Id(2, 2).array.flatten())[:8]
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        >>> list(Id(2, 2).array.flatten())[8:]
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        """
        dim = dim[0] if isinstance(dim[0], Dim) else Dim(*dim)
        array = functools.reduce(
            lambda a, x: np.tensordot(a, np.identity(x), 0)
            if a.shape else np.identity(x), dim, np.array(1))
        array = np.moveaxis(
            array, [2 * i for i in range(len(dim))], list(range(len(dim))))
        super().__init__(dim, dim, array)


class MatrixFunctor(Functor):
    """ Implements a matrix-valued rigid functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, x @ y)
    >>> F = MatrixFunctor({x: 1, y: 2}, {f: [0, 1]})
    >>> F(f)
    Matrix(dom=Dim(1), cod=Dim(2), array=[0, 1])
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_cls=Dim, ar_cls=Matrix)

    def __repr__(self):
        return super().__repr__().replace("Functor", "MatrixFunctor")

    def __call__(self, diagram):
        if isinstance(diagram, Ty):
            return sum(map(self, diagram.objects), Dim(1))
        if isinstance(diagram, Ob):
            return Dim(self.ob[Ty(Ob(diagram.name, z=0))])
        if isinstance(diagram, Cup):
            return Matrix.cups(self(diagram.dom[0]), self(diagram.dom[1]))
        if isinstance(diagram, Cap):
            return Matrix.caps(self(diagram.cod[0]), self(diagram.cod[1]))
        if isinstance(diagram, Box):
            if diagram.is_dagger:
                return self(diagram.dagger()).dagger()
            return Matrix(self(diagram.dom), self(diagram.cod),
                          self.ar[diagram])
        if not isinstance(diagram, Diagram):
            raise TypeError(messages.type_err(Diagram, diagram))

        def dim(scan):
            return len(self(scan))
        scan, array = diagram.dom, Id(self(diagram.dom)).array
        for box, off in zip(diagram.boxes, diagram.offsets):
            left = dim(scan[:off])
            if array.shape and self(box).array.shape:
                source = list(range(dim(diagram.dom) + left,
                                    dim(diagram.dom) + left + dim(box.dom)))
                target = list(range(dim(box.dom)))
                array = np.tensordot(array, self(box).array, (source, target))
            else:
                array = array * self(box).array
            source = range(len(array.shape) - dim(box.cod), len(array.shape))
            target = range(dim(diagram.dom) + left,
                           dim(diagram.dom) + left + dim(box.cod))
            array = np.moveaxis(array, list(source), list(target))
            scan = scan[:off] + box.cod + scan[off + len(box.dom):]
        return Matrix(self(diagram.dom), self(diagram.cod), array)
