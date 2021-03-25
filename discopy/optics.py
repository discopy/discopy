# -*- coding: utf-8 -*-

"""
Implements linear optical networks
"""

import numpy as np
from scipy.linalg import block_diag
from math import factorial

from discopy import cat, monoidal
from discopy.monoidal import PRO
from discopy.tensor import Dim


def npperm(M):
    """
    Numpy code for computing the permanent of a matrix,
    from https://github.com/scipy/scipy/issues/7151
    """
    n = M.shape[0]
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = M.sum(axis=0)
    p = np.prod(v)
    while (j < n - 1):
        v -= 2 * d[j] * M[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)


@monoidal.Diagram.subclass
class Diagram(monoidal.Diagram):
    """
    Linear optical network seen as a diagram of beam splitters, phase shifters
    and Mach-Zender interferometers.

    >>> mach = lambda x, y: PhaseShift(x) @ Id(PRO(1)) >> BeamSplitter(y)
    >>> assert np.allclose(MZI(0.4, 0.9).array, mach(0.4, 0.9).array)
    >>> MZI(0, 0).amp(1, [1, 0], [0, 1])
    (1+0j)
    >>> assert np.allclose((BeamSplitter(0.4) >> BeamSplitter(0.4)).array,\
                           Id(PRO(2)).array)
    >>> grid = MZI(0.5, 0.3) @ MZI(0.5, 0.3) >> MZI(0.5, 0.3) @ MZI(0.5, 0.3)
    >>> np.absolute(grid.amp(7, [1, 3, 2, 1], [1, 3, 3, 0])) ** 2
    0.01503226280870989
    """
    def __repr__(self):
        return super().__repr__().replace('Diagram', 'optics.Diagram')

    @property
    def array(self):
        """
        The array corresponding to the diagram.
        Builds a block diagonal matrix for each layer and then multiplies them
        in sequence.

        >>> MZI(0, 0).array
        array([[ 0.+0.j,  1.+0.j],
               [ 1.+0.j, -0.+0.j]])
        """
        scan, array = self.dom, np.identity(len(self.dom))
        for box, off in zip(self.boxes, self.offsets):
            left, right = len(scan[:off]), len(scan[off + len(box.dom):])
            array = np.matmul(array, block_diag(np.identity(left), box.array,
                                                np.identity(right)))
        return array

    def amp(self, n_photons, x, y, permanent=npperm):
        """
        Evaluates the amplitude of an optics.Diagram on input x and output y,
        where x and y are lists of natural numbers summing to n_photons.

        Parameters
        ----------
        n_photons : int
            Number of photons
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers
        permanent : callable, optional
            Use another function for computing the permanent
            (e.g. from thewalrus)

        >>> MZI(0, 0).amp(1, [1, 0], [0, 1])
        (1+0j)
        >>> MZI(0, np.pi).amp(1, [1, 0], [1, 0])
        (1+0j)
        """
        if sum(x) != sum(y):
            return np.array(0)
        n_modes = len(self.dom)
        assert len(x) == len(y) == n_modes
        unitary = self.array
        matrix = np.stack([unitary[:, i] for i in range(n_modes)
                          for j in range(y[i])], axis=1)
        matrix = np.stack([matrix[i] for i in range(n_modes)
                          for j in range(x[i])], axis=0)
        divisor = np.sqrt(np.prod([factorial(n) for n in x + y]))
        amp = permanent(matrix) / divisor
        return amp


class Box(Diagram, monoidal.Box):
    """
    Box in an optics.Diagram
    """
    def __init__(self, name, dom, cod, data, **params):
        if not isinstance(dom, PRO):
            raise TypeError(messages.type_err(PRO, dom))
        if not isinstance(cod, PRO):
            raise TypeError(messages.type_err(PRO, cod))
        monoidal.Box.__init__(self, name, dom, cod, data=data, **params)
        Diagram.__init__(self, dom, cod, [self], [0], layers=self.layers)

    def __repr__(self):
        return super().__repr__().replace('Box', 'optics.Box')

    @property
    def array(self):
        """ The array inside the box. """
        if isinstance(self, PhaseShift):
            return np.array(np.exp(self.data[0] * 1j))
        if isinstance(self, BeamSplitter):
            cos, sin = np.cos(self.data[0] / 2), np.sin(self.data[0] / 2)
            return np.array([sin, cos, cos, -sin]).reshape((2, 2))
        if isinstance(self, MZI):
            cos, sin = np.cos(self.data[1] / 2), np.sin(self.data[1] / 2)
            exp = np.exp(1j * self.data[0])
            return np.array([exp * sin, exp * cos, cos, -sin]).reshape((2, 2))
        return np.array(self.data).reshape(Dim(len(self.dom))
                                           @ Dim(len(self.cod)) or (1, ))


class Id(monoidal.Id, Diagram):
    """
    Identity optics.Diagram
    """
    def __init__(self, dom=PRO()):
        if isinstance(dom, int):
            dom = PRO(dom)
        monoidal.Id.__init__(self, dom)
        Diagram.__init__(self, dom, dom, [], [], layers=cat.Id(dom))


class PhaseShift(Box):
    """
    Phase shifter

    Parameters
    ----------
    phase : float
    """
    def __init__(self, phase):
        super().__init__('Phase shift', PRO(1), PRO(1), [phase])


class BeamSplitter(Box):
    """
    Beam splitter

    Parameters
    ----------
    angle : float
    """
    def __init__(self, angle):
        super().__init__('Beam splitter', PRO(2), PRO(2), [angle])


class MZI(Box):
    """
    Mach-Zender interferometer

    Parameters
    ----------
    phase, angle : float
    """
    def __init__(self, phase, angle):
        super().__init__('MZI', PRO(2), PRO(2), [phase, angle])
