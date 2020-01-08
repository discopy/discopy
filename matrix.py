# -*- coding: utf-8 -*-

"""
Implements dagger monoidal functors into matrices.

>>> n = Ty('n')
>>> Alice, Bob = Box('Alice', Ty(), n), Box('Bob', Ty(), n)
>>> loves = Box('loves', n, n)
>>> ob, ar = {n: Dim(2)}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = MatrixFunctor(ob, ar)
>>> F(Alice >> loves >> Bob.dagger())
Matrix(dom=Dim(1), cod=Dim(1), array=[1])
"""

from functools import reduce as fold
from discopy import pivotal, config
from discopy.cat import Quiver
from discopy.pivotal import Ob, Ty, Box, Diagram, PivotalFunctor

try:
    import warnings
    for msg in config.IGNORE:
        warnings.filterwarnings("ignore", message=msg)
    import jax.numpy as np
except ImportError:
    import numpy as np


class Dim(Ty):
    """ Implements dimensions as tuples of positive integers.
    Dimensions form a monoid with product @ and unit Dim(1).

    >>> Dim(1) @ Dim(2) @ Dim(3)
    Dim(2, 3)
    """
    def __init__(self, *xs):
        """
        >>> len(Dim(2, 3, 4))
        3
        >>> len(Dim(1, 2, 3))
        2
        >>> Dim(-1)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected positive integer, got -1 instead.
        """
        for x in xs:
            if not isinstance(x, int) or x < 1:
                raise ValueError("Expected positive integer, got {} instead."
                                 .format(repr(x)))
        super().__init__(*[Ob(x) for x in xs if x > 1])

    def tensor(self, other):
        """
        >>> assert Dim(1) @ Dim(2, 3) == Dim(2, 3) @ Dim(1) == Dim(2, 3)
        """
        return Dim(*[x.name for x in super().tensor(other)])

    def __add__(self, other):
        """
        >>> assert sum([Dim(1), Dim(2, 3), Dim(4)], Dim(1)) == Dim(2, 3, 4)
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

    def __repr__(self):
        """
        >>> Dim(1, 2, 3)
        Dim(2, 3)
        """
        return "Dim({})".format(', '.join(map(repr, self)) or '1')

    def __str__(self):
        """
        >>> print(Dim(1, 2, 3))
        Dim(2, 3)
        """
        return repr(self)

    def __hash__(self):
        """
        >>> dim = Dim(2, 3)
        >>> {dim: 42}[dim]
        42
        """
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
        """
        >>> Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
        Matrix(dom=Dim(2), cod=Dim(2), array=[0, 1, 1, 0])
        """
        self._array = np.array(array).reshape(dom + cod)
        super().__init__(array, dom, cod)

    @property
    def array(self):
        """
        >>> Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0]).array.shape
        (2, 2, 2)
        >>> list(Matrix(Dim(2), Dim(2), [0, 1, 1, 0]).array.flatten())
        [0, 1, 1, 0]
        """
        return self._array

    def __bool__(self):
        """
        >>> assert Matrix(Dim(1), Dim(1), [1])
        """
        return bool(self.array)

    def __repr__(self):
        """
        >>> Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
        Matrix(dom=Dim(2, 2), cod=Dim(2), array=[1, 0, 0, 1, 0, 1, 1, 0])
        """
        return "Matrix(dom={}, cod={}, array={})".format(
            self.dom, self.cod, list(self.array.flatten()))

    def __str__(self):
        """
        >>> print(Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0]))
        Matrix(dom=Dim(2, 2), cod=Dim(2), array=[1, 0, 0, 1, 0, 1, 1, 0])
        """
        return repr(self)

    def __add__(self, other):
        """
        >>> u = Matrix(Dim(2), Dim(2), [1, 0, 0, 0])
        >>> v = Matrix(Dim(2), Dim(2), [0, 0, 0, 1])
        >>> assert u + v == Id(2)
        """
        if not isinstance(other, Matrix):
            raise ValueError("Matrix expected, got {} of type {} instead."
                             .format(repr(other), type(other)))
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError("Cannot add {} and {}.".format(self, other))
        return Matrix(self.dom, self.cod, self.array + other.array)

    def __eq__(self, other):
        """
        >>> arr = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape((2, 2, 2))
        >>> m = Matrix(Dim(2, 2), Dim(2), arr)
        >>> assert m == m and np.all(m == arr)
        """
        if not isinstance(other, Matrix):
            return self.array == other
        return (self.dom, self.cod) == (other.dom, other.cod)\
            and np.all(self.array == other.array)

    def then(self, other):
        """
        >>> m = Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
        >>> assert m >> m == m >> m.dagger() == Id(2)
        """
        if not isinstance(other, Matrix):
            raise ValueError(
                "Matrix expected, got {} instead.".format(repr(other)))
        if self.cod != other.dom:
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        array = np.tensordot(self.array, other.array, len(self.cod))\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        return Matrix(self.dom, other.cod, array)

    def tensor(self, other):
        """
        >>> v = Matrix(Dim(1), Dim(2), [1, 0])
        >>> v @ v
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[1, 0, 0, 0])
        >>> v @ v.dagger()
        Matrix(dom=Dim(2), cod=Dim(2), array=[1, 0, 0, 0])
        >>> assert v @ v.dagger() == v << v.dagger()
        """
        if not isinstance(other, Matrix):
            raise ValueError(
                "Matrix expected, got {} instead.".format(repr(other)))
        dom, cod = self.dom + other.dom, self.cod + other.cod
        array = np.tensordot(self.array, other.array, 0)\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        return Matrix(dom, cod, array)

    def dagger(self):
        """
        >>> m = Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
        >>> m.dagger()
        Matrix(dom=Dim(2), cod=Dim(2, 2), array=[1, 0, 0, 1, 0, 1, 1, 0])
        >>> v = Matrix(Dim(1), Dim(2), [0, 1])
        >>> assert (v >> m.dagger()).dagger() == m >> v.dagger()
        """
        array = np.moveaxis(
            self.array, range(len(self.dom + self.cod)),
            [i + len(self.cod) if i < len(self.dom) else
             i - len(self.dom) for i in range(len(self.dom + self.cod))])
        return Matrix(self.cod, self.dom, np.conjugate(array))

    @staticmethod
    def id(x):
        """
        >>> assert Id(2) == Matrix(Dim(2), Dim(2), [1, 0, 0, 1])
        """
        return Id(x)

    @staticmethod
    def cups(x, y):
        """
        >>> assert np.all(Matrix.cups(Dim(2), Dim(2)).array == np.identity(2))
        """
        if not isinstance(x, Dim) or not isinstance(y, Dim):
            raise ValueError("Expected Dim, got {} of type {} instead."
                             .format(repr((x, y)), (type(x), type(y))))
        if x.r != y:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        return Matrix(x @ y, Dim(1), Id(x).array)

    @staticmethod
    def caps(x, y):
        """
        >>> assert np.all(Matrix.caps(Dim(2), Dim(2)).array == np.identity(2))
        """
        if not isinstance(x, Dim) or not isinstance(y, Dim):
            raise ValueError("Expected Dim, got {} of type {} instead."
                             .format(repr((x, y)), (type(x), type(y))))
        if x.r != y:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        return Matrix(Dim(1), x @ y, Id(x).array)


class AxiomError(pivotal.AxiomError):
    """
    >>> m = Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    >>> m >> m  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    discopy.matrix.AxiomError: Matrix... does not compose with Matrix...
    """


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
        array = fold(lambda a, x: np.tensordot(a, np.identity(x), 0)
                     if a.shape else np.identity(x), dim, np.array(1))
        array = np.moveaxis(
            array, [2 * i for i in range(len(dim))], list(range(len(dim))))
        super().__init__(dim, dim, array)


class MatrixFunctor(PivotalFunctor):
    """ Implements a matrix-valued pivotal functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, x @ y)
    >>> F = MatrixFunctor({x: 1, y: 2}, {f: [0, 1]})
    >>> F(f)
    Matrix(dom=Dim(1), cod=Dim(2), array=[0, 1])
    """
    def __init__(self, ob, ar):
        """
        >>> MatrixFunctor({Ty('x'): 2}, {})
        MatrixFunctor(ob={Ty('x'): Dim(2)}, ar={})
        >>> MatrixFunctor({Ty('x'): Dim(2)}, {})
        MatrixFunctor(ob={Ty('x'): Dim(2)}, ar={})
        >>> MatrixFunctor({Ty('x'): Ty('y')}, {})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected int or Dim object, got Ty('y') instead.
        """
        for x, y in ob.items():
            if isinstance(y, int):
                ob.update({x: Dim(y)})
            elif not isinstance(y, Dim):
                raise ValueError(
                    "Expected int or Dim object, got {} instead."
                    .format(repr(y)))
        super().__init__(ob, {}, Dim, Matrix)
        self._input_ar, self._ar = ar, Quiver(
            lambda box: Matrix(self(box.dom), self(box.cod), ar[box]))

    def __repr__(self):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> MatrixFunctor({x: 1, y: 3}, {})
        MatrixFunctor(ob={Ty('x'): Dim(1), Ty('y'): Dim(3)}, ar={})
        >>> MatrixFunctor({}, {Box('f', x @ x, y): list(range(3))})
        MatrixFunctor(ob={}, ar={Box('f', Ty('x', 'x'), Ty('y')): [0, 1, 2]})
        """
        return "MatrixFunctor(ob={}, ar={})".format(self.ob, self._input_ar)

    def __call__(self, diagram):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x @ x, y), Box('g', y, Ty())
        >>> ob = {x: 2, y: 3}
        >>> ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
        >>> F = MatrixFunctor(ob, ar)
        >>> list(F(f >> g).array.flatten())
        [5.0, 14.0, 23.0, 32.0]
        >>> F(f @ f.dagger()).array.shape
        (2, 2, 3, 3, 2, 2)
        >>> F(f.dagger() @ f).array.shape
        (3, 2, 2, 2, 2, 3)
        >>> list(F(f.dagger() >> f).array.flatten())
        [126.0, 144.0, 162.0, 144.0, 166.0, 188.0, 162.0, 188.0, 214.0]
        >>> list(F(g.dagger() >> g).array.flatten())
        [5]
        >>> assert F((x @ y).r) == F(x @ y).r
        >>> assert np.all(F(f.transpose_l()).array == F(f).array.transpose())
        >>> assert np.all(F(f.transpose_r()).array == F(f).array.transpose())
        """
        if isinstance(diagram, (Ty, Box)) or not isinstance(diagram, Diagram):
            return super().__call__(diagram)

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
