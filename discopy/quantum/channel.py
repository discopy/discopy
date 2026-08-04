# -*- coding: utf-8 -*-
"""
Implements classical-quantum channels.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    CQ
    Channel
    Functor


.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        C
        Q

Note
----
:class:`Channel` implements the classical-quantum processes of
Coecke and Kissinger :cite:`CoeckeKissinger17`.
Objects are given by a quantum dimension :class:`Q` (a.k.a. double wires)
and a classical dimension :class:`C` (a.k.a. single wires).
Arrows are given by arrays of the appropriate shape, see :class:`Channel`.
For example, states of type :class:`Q` are density matrices:

Example
-------
>>> from discopy.quantum import Ket, H
>>> (Ket(0) >> H).eval(mixed=True).round(1)
Channel([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dom=CQ(), cod=Q(Dim(2)))
"""

from __future__ import annotations

from discopy import frobenius, tensor
from discopy.cat import factory
from discopy.frobenius import Box
from discopy.matrix import backend
from discopy.quantum.circuit import (
    Digit, Qudit)
from discopy.quantum.gates import Discard, Measure, MixedState, Encode, Scalar
from discopy.tensor import Dim, Tensor
from discopy.utils import assert_isinstance


class CQ:
    """
    A classical-quantum dimension is a pair of dimensions
    ``classical`` and ``quantum``.

    Parameters:
        classical (Dim) : Classical dimension of the type.
        quantum (Dim) : Quantum dimension of the type.

    Example
    -------
    >>> CQ(Dim(2), Dim(3)) @ CQ(Dim(4), Dim(5))
    CQ(classical=Dim(2, 4), quantum=Dim(3, 5))
    """
    def __init__(self, classical=Dim(1), quantum=Dim(1)):
        self.classical, self.quantum = classical, quantum

    def to_dim(self) -> Dim:
        """
        The underlying dimension of the system, i.e. the classical dimension
        tensored with the square of the quantum dimension.

        Example
        -------
        >>> assert CQ(Dim(2), Dim(3)).to_dim() == Dim(2, 3, 3)
        """
        return self.classical @ self.quantum @ self.quantum

    def __eq__(self, other):
        return isinstance(other, CQ)\
            and self.classical == other.classical\
            and self.quantum == other.quantum

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return f"CQ(classical={self.classical}, quantum={self.quantum})"

    def __str__(self):
        return "CQ()" if not self.classical and not self.quantum\
            else f"Q({self.quantum})" if not self.classical\
            else f"C({self.classical})" if not self.quantum\
            else f"C({self.classical}) @ Q({self.quantum})"

    @classmethod
    def unit(cls) -> CQ:
        return cls(Dim(), quantum=Dim())

    def tensor(self, *others):
        """
        The tensor of a classical-quantum dimension with some ``others``.

        Parameters:
            others : The other types with which to tensor.
        """
        for other in others:
            assert_isinstance(other, CQ)
        classical = self.classical.tensor(*(x.classical for x in others))
        quantum = self.quantum.tensor(*(x.quantum for x in others))
        return CQ(classical, quantum)

    def __matmul__(self, other):
        return self.tensor(other) if isinstance(other, CQ) else NotImplemented

    __add__ = __matmul__

    r = l = property(lambda self: CQ(self.classical[::-1], self.quantum[::-1]))


def C(dim=Dim(1)) -> CQ:
    """
    Syntactic sugar for ``CQ(classical=dim)``, see :class:`CQ`.

    Parameters:
        dim : The dimension of the type.
    """
    return CQ(classical=dim)


def Q(dim=Dim(1)) -> CQ:
    """
    Syntactic sugar for ``CQ(quantum=dim)``, see :class:`CQ`.

    Parameters:
        dim : The dimension of the type.
    """
    return CQ(quantum=dim)


@factory
class Channel(Tensor):
    """
    A channel is a tensor with :class:`CQ` types as ``dom`` and ``cod``.

    Parameters:
        array : The array of shape ``dom.to_dim() @ cod.to_dim()``
                inside the channel.
        dom : The domain of the channel.
        cod : The codomain of the channel.
    """
    ob = CQ
    dtype = complex

    def __init__(self, array, dom: CQ, cod: CQ):
        assert_isinstance(dom, CQ)
        assert_isinstance(cod, CQ)
        super().__init__(array, dom.to_dim(), cod.to_dim())
        self.dom, self.cod = dom, cod

    def to_tensor(self) -> Tensor:
        """ The underlying tensor of a channel. """
        return Tensor[self.dtype](
            self.array, self.dom.to_dim(), self.cod.to_dim())

    @classmethod
    def id(cls, dom=CQ()) -> Channel:
        assert_isinstance(dom, CQ)
        return cls(Tensor[cls.dtype].id(dom.to_dim()).array, dom, dom)

    def then(self, other: Channel = None, *others: Channel) -> Channel:
        if other is None or others:
            return super().then(other, *others)
        assert_isinstance(other, type(self))
        array = (self.to_tensor() >> other.to_tensor()).array
        return type(self)(array, self.dom, other.cod)

    def dagger(self) -> Channel:
        return type(self)(self.to_tensor().dagger().array, self.cod, self.dom)

    @property
    def dims(self) -> list[int]:
        """
        The number of axes in each of the six blocks that make up the shape
        of a channel, i.e. the classical then twice the quantum dimension of
        its domain, then the same for its codomain.

        Example
        -------
        >>> Channel.id(C(Dim(2, 2)) @ Q(Dim(2))).dims
        [2, 1, 1, 2, 1, 1]
        """
        return [len(dim) for cq in (self.dom, self.cod)
                for dim in (cq.classical, cq.quantum, cq.quantum)]

    def tensor(self, other: Channel = None, *others: Channel) -> Channel:
        if other is None or others:
            return super().tensor(other, *others)
        assert_isinstance(other, type(self))
        left, right = self.dims, other.dims
        offsets, index = [], 0
        for size in left + right:
            offsets.append(index)
            index += size
        permutation = [
            axis for block in range(len(left))
            for start, size in [
                (offsets[block], left[block]),
                (offsets[len(left) + block], right[block])]
            for axis in range(start, start + size)]
        with backend() as np:
            array = np.transpose(
                np.tensordot(self.array, other.array, 0), permutation)
        return type(self)(array, self.dom @ other.dom, self.cod @ other.cod)

    @classmethod
    def swap(cls, left, right) -> Channel:
        array = (Tensor.swap(left.classical, right.classical)
                 @ Tensor.swap(left.quantum, right.quantum)
                 @ Tensor.swap(left.quantum, right.quantum)).array
        return cls(array, left @ right, right @ left)

    @classmethod
    def cups(cls, left, right):
        return cls.single(Tensor.cups(left.classical, right.classical))\
            @ cls.double(Tensor.cups(left.quantum, right.quantum))

    @classmethod
    def measure(cls, dim: Dim, destructive=True) -> Channel:
        """
        Measure a quantum dimension into a classical dimension.

        Parameters:
            dim : The dimension of the quantum system to measure.
            destructive : Whether the measurement discards the qubits.

        Note
        ----
        Measuring in the computational basis is the copy spider of the
        classical structure it induces, with the two legs in given by the
        double wire of ``Q(dim)``.
        """
        if not dim:
            return cls.id()
        if len(dim) > 1:
            return cls.measure(dim[:1], destructive)\
                @ cls.measure(dim[1:], destructive)
        n_legs_out = 2 if destructive else 4
        array = Tensor[cls.dtype].copy(dim, n_legs_out).array
        cod = C(dim) if destructive else C(dim) @ Q(dim)
        return cls(array, Q(dim), cod)

    @classmethod
    def encode(cls, dim: Dim, constructive=True) -> Channel:
        """
        Encode a classical dimension into a quantum dimension.

        Parameters:
            dim : The dimension of the classical system to encode.
            constructive : Whether the encoding prepares fresh qubits.
        """
        return cls.measure(dim, destructive=constructive).dagger()

    @classmethod
    def double(cls, quantum: Tensor) -> Channel:
        """
        Construct a pure quantum channel by doubling a given tensor.

        Parameters:
            quantum : The tensor from which to make a pure quantum channel.
        """
        array = (quantum @ quantum.conjugate(diagrammatic=False)).array
        return cls(array, Q(quantum.dom), Q(quantum.cod))

    @classmethod
    def single(cls, classical: Tensor) -> Channel:
        """
        Construct a pure classical channel from a given tensor.

        Parameters:
            classical : The tensor from which to make a pure classical channel.
        """
        return cls(classical.array, C(classical.dom), C(classical.cod))

    @classmethod
    def discard(cls, dom: CQ) -> Channel:
        """
        Construct the channel that traces out the quantum dimension and takes
        the marginal distribution over the classical dimension.

        Parameters:
            dom : The classical-quantum dimension to discard.
        """
        with backend() as np:
            array = np.tensordot(
                np.ones(dom.classical.inside), Tensor.id(dom.quantum).array, 0)
        return cls(array, dom, CQ())


class Functor(tensor.Functor):
    """
    A channel functor is a tensor functor into classical-quantum channels.

    Parameters:
        ob_map (dict[cat.Ob, CQ]) : The object mapping.
        ar_map (dict[cat.Box, array]) : The arrow mapping.
        dom : The domain of the functor.
        dtype : The datatype for the codomain ``Channel[dtype]``.
    """
    dom, cod = frobenius.Diagram, Channel

    def __call__(self, other):
        if isinstance(other, Digit):
            return C(Dim(other.dim))
        if isinstance(other, Qudit):
            return Q(Dim(other.dim))
        if not isinstance(other, Box):
            return frobenius.Functor.__call__(self, other)
        if isinstance(other, Discard):
            return self.cod.discard(self(other.dom))
        if isinstance(other, Measure):
            measure = self.cod.measure(
                self(other.dom).quantum, destructive=other.destructive)
            measure = measure @ self.cod.discard(C(self(other.dom).classical))\
                if other.override_bits else measure
            return measure
        if isinstance(other, (MixedState, Encode)):
            return self(other.dagger()).dagger()
        if isinstance(other, Scalar):
            scalar = other.array if other.is_mixed else abs(other.array) ** 2
            return self.cod(scalar, CQ(), CQ())
        if not other.is_mixed and other.is_classical:
            dom, cod = self(other.dom).classical, self(other.cod).classical
            return self.cod.single(
                Tensor[self.dtype](other.array, dom, cod))
        if not other.is_mixed:
            dom, cod = self(other.dom).quantum, self(other.cod).quantum
            return self.cod.double(
                Tensor[self.dtype](other.array, dom, cod))
        if hasattr(other, "array"):
            return self.cod(other.array, self(other.dom), self(other.cod))
        return frobenius.Functor.__call__(self, other)
