# -*- coding: utf-8 -*-

"""
Implements quantum gates as boxes within a circuit diagram.

>>> X = Gate('X', 1, [0, 1, 1, 0])
>>> X
Gate('X', 1, [0, 1, 1, 0])
>>> (X.dom, X.cod)
(PRO(1), PRO(1))
>>> X.array.shape
(2, 2)
"""

from functools import wraps
from discopy.moncat import Box
from discopy.matrix import Dim, Matrix
from discopy.circuit import PRO, Circuit, Id
from discopy import config

try:
    import jax.numpy as np
except ImportError:
    import numpy as np


class Gate(Box, Circuit):
    """ Implements quantum gates as boxes in a circuit diagram.

    >>> CX
    Gate('CX', 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
    """
    def __init__(self, name, n_qubits, array, data={}, _dagger=False):
        """
        >>> g = CX
        >>> assert g.dom == g.cod == PRO(2)
        """
        self._array = np.array(array).reshape(2 * n_qubits * (2, ) or 1)
        Box.__init__(self, name, PRO(n_qubits), PRO(n_qubits),
                     data=data, _dagger=_dagger)

    @property
    def array(self):
        """
        >>> list(X.array.flatten())
        [0, 1, 1, 0]
        """
        return self._array

    def __repr__(self):
        """
        >>> CX
        Gate('CX', 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        """
        if self._dagger:
            return repr(self.dagger()) + '.dagger()'
        return "Gate({}, {}, {}{})".format(
            repr(self.name), len(self.dom), list(self.array.flatten()),
            ', data=' + repr(self.data) if self.data else '')

    def __str__(self):
        """
        >>> print(CX)
        CX
        >>> print(Rx(0.25))
        Rx(0.25)
        """
        if self._dagger:
            return str(self.dagger()) + '.dagger()'
        if self.name in ['Rx', 'Rz']:
            return "{}({})".format(self.name, self.data['phase'])
        return self.name

    def dagger(self):
        """
        >>> print(CX.dagger())
        CX.dagger()
        >>> print(Rx(0.25).dagger())
        Rx(-0.25)
        >>> assert Rx(0.25).eval().dagger() == Rx(0.25).dagger().eval()
        """
        return Gate(self.name, len(self.dom), self.array,
                    data=self.data, _dagger=not self._dagger)


class Ket(Gate):
    """ Implements ket for a given bitstring.

    >>> Ket(1, 1, 0).eval()
    Matrix(dom=Dim(1), cod=Dim(2, 2, 2), array=[0, 0, 0, 0, 0, 0, 1, 0])
    """
    def __init__(self, *bitstring):
        """
        >>> g = Ket(1, 1, 0)
        """
        self.bitstring = bitstring
        Box.__init__(self, 'Ket({})'.format(', '.join(map(str, bitstring))),
                     PRO(0), PRO(len(bitstring)))

    def __repr__(self):
        """
        >>> Ket(1, 1, 0)
        Ket(1, 1, 0)
        """
        return self.name

    def dagger(self):
        """
        >>> Ket(0, 1).dagger()
        Bra(0, 1)
        """
        return Bra(*self.bitstring)

    @property
    def array(self):
        """
        >>> Ket(0).eval()
        Matrix(dom=Dim(1), cod=Dim(2), array=[1, 0])
        >>> Ket(0, 1).eval()
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[0, 1, 0, 0])
        """
        m = Matrix(Dim(1), Dim(1), [1])
        for b in self.bitstring:
            m = m @ Matrix(Dim(1), Dim(2), [0, 1] if b else [1, 0])
        return m.array


class Bra(Gate):
    """ Implements bra for a given bitstring.

    >>> Bra(1, 1, 0).eval()
    Matrix(dom=Dim(2, 2, 2), cod=Dim(1), array=[0, 0, 0, 0, 0, 0, 1, 0])
    >>> assert all((Bra(x, y, z) << Ket(x, y, z)).eval() == 1
    ...            for x in [0, 1] for y in [0, 1] for z in [0, 1])
    """
    def __init__(self, *bitstring):
        """
        >>> g = Bra(1, 1, 0)
        """
        self.bitstring = bitstring
        Box.__init__(self, 'Bra({})'.format(', '.join(map(str, bitstring))),
                     PRO(len(bitstring)), PRO(0))

    def __repr__(self):
        """
        >>> Bra(1, 1, 0)
        Bra(1, 1, 0)
        """
        return self.name

    def dagger(self):
        """
        >>> Bra(0, 1).dagger()
        Ket(0, 1)
        """
        return Ket(*self.bitstring)

    @property
    def array(self):
        """
        >>> Bra(0).eval()
        Matrix(dom=Dim(2), cod=Dim(1), array=[1, 0])
        >>> Bra(0, 1).eval()
        Matrix(dom=Dim(2, 2), cod=Dim(1), array=[0, 1, 0, 0])
        """
        m = Matrix(Dim(1), Dim(1), [1])
        for b in self.bitstring:
            m = m @ Matrix(Dim(2), Dim(1), [0, 1] if b else [1, 0])
        return m.array


class Rz(Gate):
    """
    >>> assert np.all(Rz(0).array == np.identity(2))
    >>> assert np.allclose(Rz(0.5).array, Z.array)
    >>> assert np.allclose(Rz(0.25).array, S.array)
    >>> assert np.allclose(Rz(0.125).array, T.array)
    """
    def __init__(self, phase):
        """
        >>> g = Rz(0.25)
        >>> assert g == Rz(0.25)
        >>> g.data['phase'] = 0
        >>> g
        Rz(0)
        """
        Box.__init__(self, 'Rz', PRO(1), PRO(1), data={'phase': phase})

    @property
    def name(self):
        """
        >>> assert str(Rz(0.125)) == repr(Rz(0.125)) == Rz(0.125).name
        """
        return 'Rz({})'.format(self.data['phase'])

    def __repr__(self):
        """
        >>> assert str(Rz(0.125)) == repr(Rz(0.125))
        """
        return self.name

    def dagger(self):
        """
        >>> assert Rz(0.125).dagger().eval() == Rz(0.125).eval().dagger()
        """
        return Rz(-self.data['phase'])

    @property
    def array(self):
        """
        >>> assert np.allclose(Rz(-1).array, np.identity(2))
        >>> assert np.allclose(Rz(0).array, np.identity(2))
        >>> assert np.allclose(Rz(1).array, np.identity(2))
        """
        theta = 2 * np.pi * self.data['phase']
        return np.array([[1, 0], [0, np.exp(1j * theta)]])


class Rx(Gate):
    """
    >>> assert np.all(Rx(0).array == np.identity(2))
    >>> assert np.all(np.round(Rx(0.5).array) == X.array)
    """
    def __init__(self, phase):
        """
        >>> g = Rx(0.25)
        >>> assert g == Rx(0.25)
        >>> g.data['phase'] = 0
        >>> g
        Rx(0)
        """
        Box.__init__(self, 'Rx', PRO(1), PRO(1), data={'phase': phase})

    @property
    def name(self):
        """
        >>> assert str(Rx(0.125)) == Rx(0.125).name
        """
        return 'Rx({})'.format(self.data['phase'])

    def __repr__(self):
        """
        >>> assert str(Rx(0.125)) == repr(Rx(0.125))
        """
        return self.name

    def dagger(self):
        """
        >>> assert Rx(0.125).dagger().eval() == Rx(0.125).eval().dagger()
        """
        return Rx(-self.data['phase'])

    @property
    def array(self):
        """
        >>> assert np.allclose(Rx(0).array, np.identity(2))
        >>> assert np.allclose(np.round(Rx(0.5).array), X.array)
        >>> assert np.allclose(np.round(Rx(-1).array), np.identity(2))
        >>> assert np.allclose(np.round(Rx(1).array), np.identity(2))
        >>> assert np.allclose(np.round(Rx(2).array), np.identity(2))
        """
        half_theta = np.pi * self.data['phase']
        global_phase = np.exp(1j * half_theta)
        sin, cos = np.sin(half_theta), np.cos(half_theta)
        return global_phase * np.array([[cos, -1j * sin], [-1j * sin, cos]])


def sqrt(x):
    """
    >>> sqrt(2)  # doctest: +ELLIPSIS
    Gate('sqrt(2)', 0, [1.41...])
    """
    return Gate('sqrt({})'.format(x), 0, np.sqrt(x))


SWAP = Gate('SWAP', 2, [1, 0, 0, 0,
                        0, 0, 1, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 1])
CX = Gate('CX', 2, [1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 0])
H = Gate('H', 1, 1 / np.sqrt(2) * np.array([1, 1, 1, -1]))
S = Gate('S', 1, [1, 0, 0, 1j])
T = Gate('T', 1, [1, 0, 0, np.exp(1j * np.pi / 4)])
X = Gate('X', 1, [0, 1, 1, 0])
Y = Gate('Y', 1, [0, -1j, 1j, 0])
Z = Gate('Z', 1, [1, 0, 0, -1])
