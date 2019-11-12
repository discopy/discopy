""" Implements dagger monoidal functors into matrices.
"""

import numpy as np
from functools import reduce as fold
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor


DEFAULT_TYPE = int

class Dim(Ty):
    """ Implements dimensions as tuples of positive integers.
    Dimensions form a monoid with product + and unit Dim(1).

    >>> Dim(2, 3, 4)
    Dim(2, 3, 4)
    >>> Dim(1, 2, 3)
    Dim(2, 3)
    >>> Dim(2, 3) + Dim(4)
    Dim(2, 3, 4)
    >>> len(Dim(2, 3, 4))
    3
    >>> len(Dim(1, 2, 3))
    2
    >>> assert Dim(1) + Dim(2, 3) == Dim(2, 3) + Dim(1) == Dim(2, 3)
    >>> assert Dim(2, 3)[:1] == Dim(3, 2)[1:] == Dim(2)
    >>> assert Dim(2, 3)[0] == Dim(3, 2)[1] == 2
    """
    def __init__(self, *xs):
        assert all(isinstance(x, int) and x >= 1 for x in xs)
        super().__init__(*[Ob(x) for x in xs if x > 1])

    def __add__(self, other):
        return Dim(*[x.name for x in super().__add__(other)])

    def __getitem__(self, key):  # allows to compute slices of types
        if isinstance(key, slice):
            return Dim(*[x.name for x in super().__getitem__(key)])
        return super().__getitem__(key).name

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return "Dim({})".format(', '.join(map(repr, self)) or '1')

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(repr(self))

class Matrix(Diagram):
    """ Implements a matrix with dom, cod and numpy array.

    >>> m = Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    >>> m
    Matrix(dom=Dim(2, 2), cod=Dim(2), array=[1, 0, 0, 1, 0, 1, 1, 0])
    >>> v = Matrix(Dim(2), Dim(1), [0, 1])
    >>> v
    Matrix(dom=Dim(2), cod=Dim(1), array=[0, 1])
    >>> (m >> v).array.flatten()
    array([0, 1, 1, 0])
    >>> (v @ v).array.flatten()
    array([0, 0, 0, 1])
    >>> v.dagger() @ v.dagger() >> m
    Matrix(dom=Dim(1), cod=Dim(2), array=[1, 0])
    """
    def __init__(self, dom, cod, array):
        dom, cod = Dim(*dom), Dim(*cod)
        array = np.array(array).reshape(dom + cod)
        self._dom, self._cod, self._array = dom, cod, array

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "Matrix(dom={}, cod={}, array={})".format(
                       self.dom, self.cod, list(self.array.flat))

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return self.array == other
        return (self.dom, self.cod) == (other.dom, other.cod) and np.all(
                self.array == other.array)

    def then(self, other):
        assert self.cod == other.dom
        return Matrix(self.dom, other.cod,
                      np.tensordot(self.array, other.array, len(self.cod)))

    def tensor(self, other):
        dom, cod = self.dom + other.dom, self.cod + other.cod
        array = np.tensordot(self.array, other.array, 0)
        return Matrix(dom, cod, array)

    def dagger(self):
        array = np.moveaxis(self.array, range(len(self.dom + self.cod)),
            [i + len(self.cod) if i < len(self.dom) else
             i - len(self.dom) for i in range(len(self.dom + self.cod))])
        return Matrix(self.cod, self.dom, np.conjugate(array))

    @staticmethod
    def id(dim):
        return Id(dim)

class Id(Matrix):
    """ Implements the identity matrix for a given dimension.

    >>> Id(1)
    Matrix(dom=Dim(1), cod=Dim(1), array=[1])
    >>> Id(2)
    Matrix(dom=Dim(2), cod=Dim(2), array=[1.0, 0.0, 0.0, 1.0])
    >>> Id(2).array
    array([[1., 0.],
           [0., 1.]])
    >>> Id(2).array.shape
    (2, 2)
    >>> Id(1, 2, 3)  # doctest: +ELLIPSIS
    Matrix(dom=Dim(2, 3), cod=Dim(2, 3), array=[1.0, ..., 1.0])
    >>> Id(1, 2, 3).array.shape
    (2, 3, 2, 3)
    """
    def __init__(self, *dim):
        dim = dim[0] if isinstance(dim[0], Dim) else Dim(*dim)
        array = np.moveaxis(
            fold(lambda a, x: np.tensordot(a, np.identity(x), 0), dim, 1),
            [2 * i for i in range(len(dim))], [i for i in range(len(dim))])
        super().__init__(dim, dim, array)

    def __str__(self):
        return repr(self)

class MatrixFunctor(MonoidalFunctor):
    """ Implements a matrix-valued functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x + x, y), Box('g', y, Ty())
    >>> ob = {x: 2, y: 3}
    >>> ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
    >>> F = MatrixFunctor(ob, ar)
    >>> F(f >> g).array.flatten()
    array([ 5., 14., 23., 32.])
    >>> F(f @ f.dagger()).array.shape
    (2, 2, 3, 3, 2, 2)
    >>> F(f.dagger() @ f).array.shape
    (3, 2, 2, 2, 2, 3)
    >>> F(f.dagger() >> f).array.flatten()
    array([126., 144., 162., 144., 166., 188., 162., 188., 214.])
    >>> F(g.dagger() >> g).array
    array(5)
    """
    def __call__(self, d):
        if isinstance(d, Ty):
            return Dim(*(self.ob[x] for x in d))
        elif isinstance(d, Box):
            if d._dagger: return Matrix(
                self(d.cod), self(d.dom), self.ar[d.dagger()]).dagger()
            return Matrix(self(d.dom), self(d.cod), self.ar[d])
        scan, array, dim = d.dom, Id(self(d.dom)).array, lambda t: len(self(t))
        for f, offset in d:
            n = dim(scan[:offset])
            source = range(dim(d.dom) + n, dim(d.dom) + n + dim(f.dom))
            target = range(dim(f.dom))
            array = np.tensordot(array, self(f).array, (source, target))
            source = range(len(array.shape) - dim(f.cod), len(array.shape))
            target = range(dim(d.dom) + n, dim(d.dom) + n + dim(f.cod))
            array = np.moveaxis(array, source, target)
            scan = scan[:offset] + f.cod + scan[offset + len(f.dom):]
        return Matrix(self(d.dom), self(d.cod), array)
