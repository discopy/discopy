# -*- coding: utf-8 -*-

"""
Implements dagger monoidal functors into tensors.

>>> n = Ty('n')
>>> Alice, Bob = Box('Alice', Ty(), n), Box('Bob', Ty(), n)
>>> loves = Box('loves', n, n)
>>> ob, ar = {n: 2}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = TensorFunctor(ob, ar)
>>> assert F(Alice >> loves >> Bob.dagger()) == 1
"""

import functools

from discopy import messages, monoidal, rigid
from discopy.cat import AxiomError
from discopy.monoidal import Swap
from discopy.rigid import Ob, Ty, Box, Cup, Cap, Diagram, Functor

try:  # pragma: no cover
    import warnings
    for msg in messages.IGNORE_WARNINGS:
        warnings.filterwarnings("ignore", message=msg)
    import jax.numpy as np
    def array2string(array, max_length=messages.NUMPY_THRESHOLD):
        """ array2string is not implemented in jax.numpy """
        ls = list(array)
        if len(ls) > max_length:
            ls = ls[:max_length // 2] + ["..."] + ls[1 - max_length // 2:]
        return "[{}]".format(", ".join(map(str, ls)))
    np.array2string = array2string
except ImportError:  # pragma: no cover
    import numpy as np
    from numpy import array2string as _array2string
    np.set_printoptions(threshold=messages.NUMPY_THRESHOLD)
    def array2string(array, **params):
        """ makes sure we get the same doctest with numpy and jax.numpy """
        return _array2string(array, separator=', ', **params)\
            .replace('[ ', '[').replace('  ',  ' ')
    np.array2string = array2string


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

    def tensor(self, *others):
        return Dim(*[x.name for x in super().tensor(*others)])

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


class Tensor(Box):
    """ Implements a tensor with dom, cod and numpy array.

    >>> m = Tensor(Dim(2), Dim(2), [0, 1, 1, 0])
    >>> v = Tensor(Dim(1), Dim(2), [0, 1])
    >>> v >> m >> v.dagger()
    Tensor(dom=Dim(1), cod=Dim(1), array=[0])
    """
    def __init__(self, dom, cod, array):
        self._array = np.array(array).reshape(dom + cod)
        super().__init__("Tensor", dom, cod)

    @property
    def array(self):
        """ Numpy array. """
        return self._array

    def __bool__(self):
        return bool(self.array)

    def __repr__(self):
        return "Tensor(dom={}, cod={}, array={})".format(
            self.dom, self.cod,
            np.array2string(self.array.flatten()))

    def __str__(self):
        return repr(self)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(messages.type_err(Tensor, other))
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        return Tensor(self.dom, self.cod, self.array + other.array)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return self.array == other
        return (self.dom, self.cod) == (other.dom, other.cod)\
            and np.all(self.array == other.array)

    def then(self, *others):
        if len(others) != 1:
            return monoidal.Diagram.then(self, *others)
        other = others[0]
        if not isinstance(other, Tensor):
            raise TypeError(messages.type_err(Tensor, other))
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        array = np.tensordot(self.array, other.array, len(self.cod))\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        return Tensor(self.dom, other.cod, array)

    def tensor(self, *others):
        if len(others) != 1:
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]
        if not isinstance(other, Tensor):
            raise TypeError(messages.type_err(Tensor, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        array = np.tensordot(self.array, other.array, 0)\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        source = range(len(dom @ cod))
        target = [
            i if i < len(self.dom) or i >= len(self.dom @ self.cod @ other.dom)
            else i - len(self.cod) if i >= len(self.dom @ self.cod)
            else i + len(other.dom) for i in source]
        return Tensor(dom, cod, np.moveaxis(array, source, target))

    def dagger(self):
        array = np.moveaxis(
            self.array, range(len(self.dom + self.cod)),
            [i + len(self.cod) if i < len(self.dom) else
             i - len(self.dom) for i in range(len(self.dom + self.cod))])
        return Tensor(self.cod, self.dom, np.conjugate(array))

    @staticmethod
    def id(x):
        return Id(x)

    @staticmethod
    def cups(left, right):
        return rigid.cups(
            left, right, ar_factory=Tensor,
            cup_factory=lambda left, right:
                Tensor(left @ right, Dim(1), Id(left).array))

    @staticmethod
    def caps(left, right):
        return Tensor.cups(left, right).dagger()

    @staticmethod
    def swap(left, right):
        array = Id(left @ right).array
        source = range(len(left @ right), 2 * len(left @ right))
        target = [i + len(right) if i < len(left @ right @ left)
                  else i - len(left) for i in source]
        return Tensor(left @ right, right @ left,
                      np.moveaxis(array, source, target))

    def transpose(self, left=False):
        """
        Returns the algebraic transpose.

        Note
        ----
        This is *not* the same as the diagrammatic transpose for complex dims.
        """
        return Tensor(self.cod[::-1], self.dom[::-1], self.array.transpose())

    def conjugate(self):
        """ Returns the conjugate of a tensor. """
        return Tensor(self.dom, self.cod, np.conjugate(self.array))

    def round(self, decimals=0):
        """ Rounds the entries of a tensor up to a number of decimals. """
        return Tensor(self.dom, self.cod,
                      np.around(self.array, decimals=decimals))

    @staticmethod
    def zeros(dom, cod):
        """
        Returns the zero tensor of a given shape.

        Examples
        --------
        >>> assert Tensor.zeros(Dim(2), Dim(2))\\
        ...     == Tensor(Dim(2), Dim(2), [0, 0, 0, 0])
        """
        return Tensor(dom, cod, np.zeros(dom + cod))


class Id(Tensor):
    """ Implements the identity tensor for a given dimension.

    >>> Id(2, 2)  # doctest: +ELLIPSIS
    Tensor(dom=Dim(2, 2), cod=Dim(2, 2), array=[...])
    >>> assert Id(2, 2) == Id(2) @ Id(2)
    """
    def __init__(self, *dim):
        dim = dim[0] if isinstance(dim[0], Dim) else Dim(*dim)
        super().__init__(
            dim, dim, np.identity(functools.reduce(int.__mul__, dim, 1)))


class TensorFunctor(Functor):
    """ Implements a tensor-valued rigid functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, x @ y)
    >>> F = TensorFunctor({x: 1, y: 2}, {f: [0, 1]})
    >>> F(f)
    Tensor(dom=Dim(1), cod=Dim(2), array=[0, 1])
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=Dim, ar_factory=Tensor)

    def __repr__(self):
        return super().__repr__().replace("Functor", "TensorFunctor")

    def __call__(self, diagram):
        if isinstance(diagram, monoidal.Ty):
            return sum(map(self, diagram.objects), Dim(1))
        if isinstance(diagram, Ob) and not diagram.z:
            result = self.ob[Ty(diagram.name)]
            return result if isinstance(result, Dim) else Dim(result)
        if isinstance(diagram, monoidal.Ob):
            return super().__call__(diagram)
        if isinstance(diagram, Cup):
            return Tensor.cups(self(diagram.dom[0]), self(diagram.dom[1]))
        if isinstance(diagram, Cap):
            return Tensor.caps(self(diagram.cod[0]), self(diagram.cod[1]))
        if isinstance(diagram, monoidal.Box) and not isinstance(diagram, Swap):
            if diagram.is_dagger:
                return self(diagram.dagger()).dagger()
            return Tensor(self(diagram.dom), self(diagram.cod),
                          self.ar[diagram])
        if not isinstance(diagram, monoidal.Diagram):
            raise TypeError(messages.type_err(Diagram, diagram))

        def dim(scan):
            return len(self(scan))
        scan, array = diagram.dom, Id(self(diagram.dom)).array
        for box, off in zip(diagram.boxes, diagram.offsets):
            if isinstance(box, Swap):
                source = range(
                    dim(diagram.dom @ scan[:off]),
                    dim(diagram.dom @ scan[:off] @ box.dom))
                target = [
                    i + dim(box.right)
                    if i < dim(diagram.dom @ scan[:off]) + dim(box.left)
                    else i - dim(box.left) for i in source]
                array = np.moveaxis(array, list(source), list(target))
                scan = scan[:off] + box.cod + scan[off + len(box.dom):]
                continue
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
        return Tensor(self(diagram.dom), self(diagram.cod), array)
