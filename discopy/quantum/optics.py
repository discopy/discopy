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

    >>> assert np.allclose((BeamSplitter(0.4) >> BeamSplitter(0.4)).array,\
                           Id(PRO(2)).array)
    >>> grid = MZI(0.5, 0.3) @ MZI(0.5, 0.3) >> Id(1) @ MZI(0.5, 0.3) @ Id(1)
    >>> np.absolute(grid.amp([1, 3, 2, 1], [1, 3, 3, 0]))
    0.06492701974296845
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

    def amp(self, x, y, permanent=npperm):
        """
        Evaluates the amplitude of an optics.Diagram on input x and output y,
        where x and y are lists of natural numbers summing to n_photons.

        Parameters
        ----------
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers
        permanent : callable, optional
            Use another function for computing the permanent
            (e.g. from thewalrus)

        >>> network = PhaseShift(0.4) @ Id(2) @ PhaseShift(0.5)\
                      >> MZI(0.4, 0.3) @ MZI(0.2, 0.5)
        >>> amplitude = network.amp([1, 0, 0, 1], [1, 0, 1, 0])
        >>> amplitude
        (-1.53080849893419e-17+4.711344115737722e-17j)
        >>> probability = np.abs(amplitude) ** 2
        >>> probability
        2.454013803730561e-33
        """
        if sum(x) != sum(y):
            return 0
        n_modes = len(self.dom)
        matrix = np.stack([self.array[:, i] for i in range(n_modes)
                          for _ in range(y[i])], axis=1)
        matrix = np.stack([matrix[i] for i in range(n_modes)
                          for _ in range(x[i])], axis=0)
        divisor = np.sqrt(np.prod([factorial(n) for n in x + y]))
        return permanent(matrix) / divisor

    def eval(self, n_photons, permanent=npperm):
        """
        >>> for i, _ in enumerate(occupation_numbers(3, 2)): assert np.isclose(
        ...       sum(np.absolute(MZI(0.2, 0.4).eval(3)[0])**2), 1)
        """
        basis = occupation_numbers(n_photons, len(self.dom))
        matrix = np.zeros(dtype=complex, shape=(len(basis), len(basis)))
        for i, x in enumerate(basis):
            for j, y in enumerate(basis):
                matrix[i, j] = self.amp(x, y, permanent=permanent)
        return matrix


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
        return np.array(self.data).reshape(
            Dim(len(self.dom)) @ Dim(len(self.cod)) or (1, ))


class Id(monoidal.Id, Diagram):
    """
    Identity optics.Diagram
    """
    def __init__(self, dom=PRO()):
        if isinstance(dom, int):
            dom = PRO(dom)
        monoidal.Id.__init__(self, dom)
        Diagram.__init__(self, dom, dom, [], [], layers=cat.Id(dom))


Diagram.id = Id


class PhaseShift(Box):
    """
    Phase shifter

    Parameters
    ----------
    phase : float
    """
    def __init__(self, phase):
        self.phase = phase
        super().__init__('Phase shift', PRO(1), PRO(1), [phase])

    @property
    def array(self):
        return np.array(np.exp(2j * np.pi * self.phase))

    def dagger(self):
        return PhaseShift(-self.phase)


class BeamSplitter(Box):
    """
    Beam splitter

    Parameters
    ----------
    angle : float

    >>> y = BeamSplitter(0.4)
    >>> assert np.allclose((y>>y).eval(2), Id(2).eval(2))
    >>> assert y == y.dagger()
    >>> comp = (y @ y >> Id(1) @ y @ Id(1)) >> (y @ y >> Id(1) @ y @ Id(1)
    ...   ).dagger()
    >>> assert np.allclose(comp.eval(2), Id(4).eval(2))
    """
    def __init__(self, angle):
        self.angle = angle
        super().__init__('Beam splitter', PRO(2), PRO(2), [angle])

    @property
    def array(self):
        cos, sin = np.cos(np.pi * self.angle), np.sin(np.pi * self.angle)
        return np.array([sin, cos, cos, -sin]).reshape((2, 2))

    def dagger(self):
        return BeamSplitter(self.angle)


class MZI(Box):
    """
    Mach-Zender interferometer

    Parameters
    ----------
    phase, angle : float

    >>> MZI(0, 0).amp([1, 0], [0, 1])
    (1+0j)
    >>> MZI(0, 0).amp([1, 0], [1, 0])
    0j
    >>> mach = lambda x, y: PhaseShift(x) @ Id(1) >> BeamSplitter(y)
    >>> assert np.allclose(MZI(0.4, 0.9).eval(4), mach(0.4, 0.9).eval(4))
    >>> assert np.allclose((MZI(0.4, 0.9) >> MZI(0.4, 0.9).dagger()).eval(3),
    ...                     Id(2).eval(3))
    """
    def __init__(self, phase, angle):
        self.phase, self.angle = phase, angle
        super().__init__('MZI', PRO(2), PRO(2), [phase, angle])

    @property
    def array(self):
        cos, sin = np.cos(np.pi * self.angle), np.sin(np.pi * self.angle)
        exp = np.exp(2j * np.pi * self.phase)
        return np.array([exp * sin, exp * cos, cos, -sin]).reshape((2, 2))

    def dagger(self):
        return BeamSplitter(self.angle) >> PhaseShift(-self.phase) @ Id(1)


class Functor(monoidal.Functor):
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=PRO, ar_factory=Diagram)


def occupation_numbers(n_photons, m_modes):
    """
    Returns vectors of occupation numbers for n_photons in m_modes.

    >>> occupation_numbers(3, 2)
    [[3, 0], [2, 1], [1, 2], [0, 3]]
    >>> occupation_numbers(2, 3)
    [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
    """
    if m_modes <= 1:
        return m_modes * [[n_photons]]
    return [[head] + tail for head in range(n_photons, -1, -1)
            for tail in occupation_numbers(n_photons - head, m_modes - 1)]
