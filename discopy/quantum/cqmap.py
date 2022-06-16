# -*- coding: utf-8 -*-
"""
Implements classical-quantum maps.

:class:`CQMap` implements the classical-quantum processes of
*Picturing Quantum Processes*, Coecke and Kissinger (2018).
Objects are given by a quantum dimension :class:`Q` (a.k.a. double wires)
and a classical dimension :class:`C` (a.k.a. single wires).
Arrows are given by arrays of the appropriate shape, see :class:`CQMap`.
For example, states of type :class:`Q` are density matrices:

>>> from discopy.quantum import Ket, H
>>> (Ket(0) >> H).eval(mixed=True).round(1)
CQMap(dom=CQ(), cod=Q(Dim(2)), array=[0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
"""

from discopy import monoidal, rigid, messages, tensor
from discopy.cat import AxiomError
from discopy.rigid import Ob, Ty, Diagram
from discopy.tensor import Dim, Tensor
from discopy.quantum.circuit import (
    bit, qubit, Digit, Qudit,
    Box, Sum, Swap, Discard, Measure, MixedState, Encode)
from discopy.quantum.gates import Scalar


class CQ(Ty):
    """
    Implements the dimensions of classical-quantum systems.

    Parameters
    ----------
    classical : :class:`discopy.tensor.Dim`
        Classical dimension.
    quantum : :class:`discopy.tensor.Dim`
        Quantum dimension.


    Note
    ----

    In the category of monoids, :class:`CQ` is the product of :class:`C` and
    :class:`Q`, which are both isomorphic to :class:`discopy.tensor.Dim`.

    Examples
    --------
    >>> CQ(Dim(2), Dim(2))
    C(Dim(2)) @ Q(Dim(2))
    >>> CQ(Dim(2), Dim(2)) @ CQ(Dim(2), Dim(2))
    C(Dim(2, 2)) @ Q(Dim(2, 2))
    """
    def __init__(self, classical=Dim(1), quantum=Dim(1)):
        self.classical, self.quantum = classical, quantum
        types = [Ob("C({})".format(dim)) for dim in classical]\
            + [Ob("Q({})".format(dim)) for dim in quantum]
        super().__init__(*types)

    def __repr__(self):
        if not self:
            return "CQ()"
        if not self.classical:
            return "Q({})".format(repr(self.quantum))
        if not self.quantum:
            return "C({})".format(repr(self.classical))
        return "C({}) @ Q({})".format(repr(self.classical), repr(self.quantum))

    def __str__(self):
        return repr(self)

    def tensor(self, *others):
        classical = self.classical.tensor(*(x.classical for x in others))
        quantum = self.quantum.tensor(*(x.quantum for x in others))
        return CQ(classical, quantum)

    @property
    def l(self):
        return CQ(self.classical[::-1], self.quantum[::-1])

    @property
    def r(self):
        return self.l


class C(CQ):
    """
    Implements the classical dimension of a classical-quantum system,
    see :class:`CQ`.
    """
    def __init__(self, dim=Dim(1)):
        super().__init__(dim, Dim(1))


class Q(CQ):
    """
    Implements the quantum dimension of a classical-quantum system,
    see :class:`CQ`.
    """
    def __init__(self, dim=Dim(1)):
        super().__init__(Dim(1), dim)


class CQMap(Tensor):
    """
    Implements classical-quantum maps.

    Parameters
    ----------
    dom : :class:`CQ`
        Domain.
    cod : :class:`CQ`
        Codomain.
    array : list, optional
        Array of size :code:`product(utensor.dom @ utensor.cod)`.
    utensor : :class:`discopy.tensor.Tensor`, optional
        Underlying tensor with domain
        :code:`dom.classical @ dom.quantum ** 2` and codomain
        :code:`cod.classical @ cod.quantum ** 2``.
    """
    @property
    def utensor(self):
        """ Underlying tensor. """
        return Tensor(self._udom, self._ucod, self.array)

    def __init__(self, dom, cod, array=None, utensor=None):
        if array is None and utensor is None:
            raise ValueError("One of array or utensor must be given.")
        if utensor is None:
            udom = dom.classical @ dom.quantum @ dom.quantum
            ucod = cod.classical @ cod.quantum @ cod.quantum
        else:
            udom, ucod = utensor.dom, utensor.cod
        super().__init__(udom, ucod, utensor.array if array is None else array)
        self._dom, self._cod, self._udom, self._ucod = dom, cod, udom, ucod

    def __repr__(self):
        return super().__repr__().replace("Tensor", "CQMap")

    def __add__(self, other):
        if other == 0:
            return self
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        return CQMap(self.dom, self.cod, self.array + other.array)

    def __radd__(self, other):
        return self.__add__(other)

    @staticmethod
    def id(dom=CQ()):
        utensor = Tensor.id(dom.classical @ dom.quantum @ dom.quantum)
        return CQMap(dom, dom, utensor=utensor)

    def then(self, *others):
        if len(others) != 1:
            return monoidal.Diagram.then(self, *others)
        other, = others
        return CQMap(
            self.dom, other.cod, utensor=self.utensor >> other.utensor)

    def dagger(self):
        return CQMap(self.cod, self.dom, utensor=self.utensor.dagger())

    def tensor(self, *others):
        if len(others) != 1:
            return monoidal.Diagram.tensor(self, *others)
        other, = others
        f = rigid.Box('f', Ty('c00', 'q00', 'q00'), Ty('c10', 'q10', 'q10'))
        g = rigid.Box('g', Ty('c01', 'q01', 'q01'), Ty('c11', 'q11', 'q11'))
        above = Diagram.id(f.dom[:1] @ g.dom[:1] @ f.dom[1:2])\
            @ Diagram.swap(g.dom[1:2], f.dom[2:]) @ Diagram.id(g.dom[2:])\
            >> Diagram.id(f.dom[:1]) @ Diagram.swap(g.dom[:1], f.dom[1:])\
            @ Diagram.id(g.dom[1:])
        below =\
            Diagram.id(f.cod[:1]) @ Diagram.swap(f.cod[1:], g.cod[:1])\
            @ Diagram.id(g.cod[1:])\
            >> Diagram.id(f.cod[:1] @ g.cod[:1] @ f.cod[1:2])\
            @ Diagram.swap(f.cod[2:], g.cod[1:2]) @ Diagram.id(g.cod[2:])
        diagram2tensor = tensor.Functor(
            ob={Ty("{}{}{}".format(a, b, c)):
                z.__getattribute__(y).__getattribute__(x)
                for a, x in zip(['c', 'q'], ['classical', 'quantum'])
                for b, y in zip([0, 1], ['dom', 'cod'])
                for c, z in zip([0, 1], [self, other])},
            ar={f: self.utensor.array, g: other.utensor.array})
        return CQMap(self.dom @ other.dom, self.cod @ other.cod,
                     utensor=diagram2tensor(above >> f @ g >> below))

    @staticmethod
    def swap(left, right):
        utensor = Tensor.swap(left.classical, right.classical)\
            @ Tensor.swap(left.quantum, right.quantum)\
            @ Tensor.swap(left.quantum, right.quantum)
        return CQMap(left @ right, right @ left, utensor=utensor)

    @staticmethod
    def measure(dim, destructive=True):
        """ Measure a quantum dimension into a classical dimension. """
        if not dim:
            return CQMap(CQ(), CQ(), Tensor.np.array(1))
        if len(dim) == 1:
            if destructive:
                array = Tensor.np.array([
                    int(i == j == k)
                    for i in range(dim[0])
                    for j in range(dim[0])
                    for k in range(dim[0])])
                return CQMap(Q(dim), C(dim), array)
            array = Tensor.np.array([
                int(i == j == k == l == m)
                for i in range(dim[0])
                for j in range(dim[0])
                for k in range(dim[0])
                for l in range(dim[0])
                for m in range(dim[0])])
            return CQMap(Q(dim), C(dim) @ Q(dim), array)
        return CQMap.measure(dim[:1], destructive=destructive)\
            @ CQMap.measure(dim[1:], destructive=destructive)

    @staticmethod
    def encode(dim, constructive=True):
        """ Encode a classical dimension into a quantum dimension. """
        return CQMap.measure(dim, destructive=constructive).dagger()

    @staticmethod
    def double(utensor):
        """ Takes a tensor, returns a pure quantum CQMap. """
        density = (utensor.conjugate(diagrammatic=False) @ utensor).array
        return CQMap(Q(utensor.dom), Q(utensor.cod), density)

    @staticmethod
    def classical(utensor):
        """ Takes a tensor, returns a classical CQMap. """
        return CQMap(C(utensor.dom), C(utensor.cod), utensor.array)

    @staticmethod
    def discard(dom):
        """ Discard a quantum dimension or take the marginal distribution. """
        array = Tensor.np.tensordot(
            Tensor.np.ones(dom.classical), Tensor.id(dom.quantum).array, 0)
        return CQMap(dom, CQ(), array)

    @staticmethod
    def cups(left, right):
        return CQMap.classical(Tensor.cups(left.classical, right.classical))\
            @ CQMap.double(Tensor.cups(left.quantum, right.quantum))

    @staticmethod
    def caps(left, right):
        return CQMap.cups(left, right).dagger()

    def round(self, decimals=0):
        """ Rounds the entries of a CQMap up to a number of decimals. """
        return CQMap(self.dom, self.cod, utensor=self.utensor.round(decimals))


class Functor(rigid.Functor):
    """
    Functors from :class:`Circuit` into :class:`CQMap`.
    """
    def __init__(self, ob=None, ar=None):
        self.__ob, self.__ar = ob or {}, ar or {}
        super().__init__(self._ob, self._ar, ob_factory=CQ, ar_factory=CQMap)

    def __repr__(self):
        return "cqmap.Functor(ob={}, ar={})".format(self.__ob, self.__ar)

    def _ob(self, typ):
        """ Overrides the input mapping on objects for Digit and Qudit. """
        obj, = typ
        if isinstance(obj, Digit):
            return C(Dim(obj.dim))
        if isinstance(obj, Qudit):
            return Q(Dim(obj.dim))
        return self.__ob[typ]

    def _ar(self, box):
        """ Overrides the input mapping on arrows. """
        if isinstance(box, Discard):
            return CQMap.discard(self(box.dom))
        if isinstance(box, Measure):
            measure = CQMap.measure(
                self(box.dom).quantum, destructive=box.destructive)
            measure = measure @ CQMap.discard(self(box.dom).classical)\
                if box.override_bits else measure
            return measure
        if isinstance(box, (MixedState, Encode)):
            return self(box.dagger()).dagger()
        if isinstance(box, Scalar):
            scalar = box.array if box.is_mixed else abs(box.array) ** 2
            return CQMap(CQ(), CQ(), scalar)
        if not box.is_mixed and box.classical:
            return CQMap(self(box.dom), self(box.cod), box.array)
        if not box.is_mixed:
            dom, cod = self(box.dom).quantum, self(box.cod).quantum
            return CQMap.double(Tensor(dom, cod, box.array))
        if hasattr(box, "array"):
            return CQMap(self(box.dom), self(box.cod), box.array)
        return self.__ar[box]
