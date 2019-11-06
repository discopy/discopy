""" Implements dagger monoidal functors into matrices.

>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f, g = Box('f', x, x + y), Box('g', y + z, w)
>>> ob = {x: 1, y: 2, z: 3, w: 4}
>>> F0 = NumpyFunctor(ob, dict())
>>> F = NumpyFunctor(ob, {a: np.zeros(F0(a.dom) + F0(a.cod)) for a in [f, g]})
>>> assert F(f.dagger()).shape == tuple(F(f.cod) + F(f.dom))
"""

import numpy as np
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor


class Dim(Ty):
    """ Implements dimensions as tuples of integers strictly greater than 1.

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
        assert all(isinstance(x, int) for x in xs)
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
        return "Dim({})".format(', '.join(map(repr, self)))

    def __str__(self):
        return "({}, )".format(', '.join(map(str, self)) or '1')

    def __hash__(self):
        return hash(repr(self))

class Matrix(Diagram):
    """ Implements a typed matrix.

    >>> m = Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
    >>> v = Matrix(Dim(2), Dim(1), [0, 1])
    >>> (m >> v).array.flatten()
    array([1, 0])
    >>> (v @ v).array.flatten()
    array([0, 0, 0, 1])
    """
    def __init__(self, dom, cod, array, dtype=np.dtype('int')):
        dom, cod = Dim(*dom), Dim(*cod)
        array = np.array(array, dtype=dtype).reshape(dom + cod)
        self._dom, self._cod, self._array, self._dtype = dom, cod, array, dtype

    @property
    def array(self):
        return self._array

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return "Matrix(dom={}, cod={}, array={}, dtype={})".format(
            self.dom, self.cod, list(self.array.flat), repr(self.dtype))

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
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
        return Matrix(self.cod, self.dom, array)

    def id(self, dim):
        return Id(dim)

class Id(Matrix):
    """ Implements the identity matrix for a given dimension. """
    def __init__(self, *dim):
        if isinstance(dim[0], Dim):
            dim = dim[0]
        else:
            dim = Dim(*dim)
        array = 1
        for x in dim:
            array = np.tensordot(array, np.identity(x), 0)
        array = np.moveaxis(array,
            [2 * i for i in range(len(dim))],
            [i for i in range(len(dim))])  # bureaucracy!
        super().__init__(dim, dim, array)

    def __repr__(self):
        return "Id{}".format(self.dom)

    def __str__(self):
        return repr(self)

class NumpyFunctor(MonoidalFunctor):
    """ Implements a matrix-valued functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x + x, y), Box('g', y, Ty())
    >>> ob = {x: 2, y: 3}
    >>> ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
    >>> F = NumpyFunctor(ob, ar)
    >>> F(f >> g).flatten()
    array([ 5, 14, 23, 32])
    >>> F(f @ f.dagger()).shape
    (2, 2, 3, 3, 2, 2)
    >>> F(f.dagger() @ f).shape
    (3, 2, 2, 2, 2, 3)
    """
    def __call__(self, d):
        if isinstance(d, Ob):
            return int(self.ob[d])
        elif isinstance(d, Ty):
            return Dim(*(self.ob[x] for x in d))
        elif isinstance(d, Box):
            arr = np.array(self.ar[d.name])
            if d._dagger:
                return Matrix(self(d.cod), self(d.dom), arr.flat).dagger().array
            else:
                return arr.reshape(self(d.dom) + self(d.cod))
        scan = d.dom
        arr = Id(self(scan)).array
        for f, offset in d:
            dim = lambda t: len(self(t))
            n = dim(scan[:offset])
            source = range(dim(d.dom) + n, dim(d.dom) + n + dim(f.dom))
            target = range(dim(f.dom))
            import pdb; pdb.set_trace()
            arr = np.tensordot(arr, self(f), (source, target))

            source = range(len(arr.shape) - dim(f.cod), len(arr.shape))
            target = range(dim(d.dom) + n, dim(d.dom) + n + dim(f.cod))
            arr = np.moveaxis(arr, source, target)  # more bureaucracy!
            scan = scan[:offset] + f.cod + scan[offset + len(f.cod):]
        return arr
