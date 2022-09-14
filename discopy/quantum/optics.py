# -*- coding: utf-8 -*-

"""
This module is an implementation of linear optical circuits as diagrams built
from beam splitters, phases and Mach-Zender interferometers.

One may compute the bosonic statistics of these devices using
:py:meth:`.Diagram.indist_prob` and the statistics for distinguishable or
partially distinguishable particles using :py:meth:`.Diagram.dist_prob` and
:py:meth:`.Diagram.pdist_prob`. Amplitudes for pairs of input and output
occupation numbers are computed using :py:meth:`.Diagram.amp` and the full
matrix of amplitudes over occupation numbers is obtained using
:py:meth:`.Diagram.eval`.

One may also use the QPath calculus as defined in
https://arxiv.org/abs/2204.12985. The functor :py:func:`optics2path` decomposes
linear optical circuits into QPath diagrams. The functor :py:func:`zx2path`
turns instances of :py:class:`.zx.Diagram` into QPath diagrams
via the dual-rail encoding.

Example
-------

>>> from discopy.quantum.optics import zx2path
>>> from discopy.quantum.zx import Z
>>> from discopy import drawing
>>> drawing.equation(Z(2, 1), zx2path(Z(2, 1)), symbol='->',
...                  draw_type_labels=False, figsize=(6, 4),
...                  path='docs/_static/imgs/optics-fusion.png')

.. image:: ../_static/imgs/optics-fusion.png
    :align: center
"""

from itertools import permutations
from math import factorial

import numpy as np

from discopy import cat, messages, monoidal
from discopy.matrix import Matrix
from discopy.monoidal import PRO
from discopy.quantum.gates import format_number
from discopy.rewriting import InterchangerError


def occupation_numbers(n_photons, m_modes):
    """
    Returns vectors of occupation numbers for n_photons in m_modes.

    Example
    -------
    >>> occupation_numbers(3, 2)
    [[3, 0], [2, 1], [1, 2], [0, 3]]
    >>> occupation_numbers(2, 3)
    [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
    """
    if m_modes <= 1:
        return m_modes * [[n_photons]]
    return [[head] + tail for head in range(n_photons, -1, -1)
            for tail in occupation_numbers(n_photons - head, m_modes - 1)]


def npperm(M):
    """
    Numpy code for computing the permanent of a matrix,
    from https://github.com/scipy/scipy/issues/7151.
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

    Example
    -------
    >>> BS >> BS
    optics.Diagram(dom=PRO(2), cod=PRO(2), boxes=[BS, BS], offsets=[0, 0])
    """
    def __repr__(self):
        return super().__repr__().replace('Diagram', 'optics.Diagram')

    @property
    def array(self):
        """
        The array corresponding to the diagram.
        Builds a block diagonal matrix for each layer and then multiplies them
        in sequence.

        Example
        -------
        >>> np.shape(to_matrix(BS).array)
        (2, 2)
        >>> np.shape(to_matrix(BS >> BS).array)
        (2, 2)
        >>> np.shape(to_matrix(BS @ BS @ BS).array)
        (6, 6)
        >>> assert np.allclose(
        ...     to_matrix(MZI(0, 0)).array, np.array([[0, 1], [1, 0]]))
        >>> assert np.allclose(to_matrix(MZI(0, 0) >> MZI(0, 0)).array,
        ...                    to_matrix(Id(2)).array)
        """
        return to_matrix(self).array

    def amp(self, x, y, permanent=npperm):
        """
        Evaluates the amplitude of an optics.Diagram on input x and output y,
        when sending indistinguishable photons.

        Parameters
        ----------
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers
        permanent : callable, optional
            Use another function for computing the permanent
            or set permanent = np.determinant to compute fermionic statistics

        Example
        -------
        >>> network = MZI(0.2, 0.4) @ MZI(0.2, 0.4)\
                      >> Id(1) @ MZI(0.2, 0.4) @ Id(1)
        >>> amplitude = network.amp([1, 0, 0, 1], [1, 0, 1, 0])
        >>> probability = np.abs(amplitude) ** 2
        >>> assert probability > 0.05
        """
        if sum(x) != sum(y):
            raise ValueError("Number of photons in != Number of photons out.")
        n_modes = len(self.dom)
        matrix = np.stack([self.array[:, i] for i in range(n_modes)
                          for _ in range(y[i])], axis=1)
        matrix = np.stack([matrix[i] for i in range(n_modes)
                          for _ in range(x[i])], axis=0)
        divisor = np.sqrt(np.prod([factorial(n) for n in x + y]))
        return permanent(matrix) / divisor

    def eval(self, n_photons, permanent=npperm):
        """
        Evaluates the matrix acting on the Fock space given number of photons.

        Parameters
        ----------
        n_photons : int
            Number of photons
        permanent : callable, optional
            Use another function for computing the permanent
            (e.g. from thewalrus)

        Example
        -------
        >>> for i, _ in enumerate(occupation_numbers(3, 2)): assert np.isclose(
        ...       sum(np.absolute(MZI(0.2, 0.4).eval(3)[i])**2), 1)
        >>> network = MZI(0.2, 0.4) @ Id(1) >> Id(1) @ MZI(0.2, 0.4)
        >>> for i, _ in enumerate(occupation_numbers(2, 3)): assert np.isclose(
        ...       sum(np.absolute(network.eval(2)[i])**2), 1)
        """
        basis = occupation_numbers(n_photons, len(self.dom))
        matrix = np.zeros(dtype=complex, shape=(len(basis), len(basis)))
        for i, x in enumerate(basis):
            for j, y in enumerate(basis):
                matrix[i, j] = self.amp(x, y, permanent=permanent)
        return matrix

    def indist_prob(self, x, y, permanent=npperm):
        """
        Evaluates the probability for indistinguishable bosons by taking
        the born rule of the amplitude.

        Parameters
        ----------
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers
        permanent : callable, optional
            Use another function for computing the permanent
            (e.g. from thewalrus)

        Example
        -------
        >>> box = MZI(0.2, 0.6)
        >>> assert np.isclose(sum([box.indist_prob([3, 0], y)
        ...                        for y in occupation_numbers(3, 2)]), 1)
        >>> network = box @ box @ box >> Id(1) @ box @ box @ Id(1)
        >>> assert np.isclose(sum([network.indist_prob([0, 1, 0, 1, 1, 1], y)
        ...                        for y in occupation_numbers(4, 6)]), 1)
        """
        return np.absolute(self.amp(x, y, permanent=permanent)) ** 2

    def dist_prob(self, x, y, permanent=npperm):
        """
        Evaluates probability of an optics.Diagram for input x and output y,
        when sending distinguishable particles.

        Parameters
        ----------
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers
        permanent : callable, optional
            Use another function for computing the permanent
            (e.g. from thewalrus)

        Example
        -------
        >>> box = MZI(1.2, 0.6)
        >>> assert np.isclose(sum([box.dist_prob([3, 0], y)
        ...                        for y in occupation_numbers(3, 2)]), 1)
        >>> network = box @ box @ box >> Id(1) @ box @ box @ Id(1)
        >>> assert np.isclose(sum([network.dist_prob([0, 1, 0, 1, 1, 1], y)
        ...                        for y in occupation_numbers(4, 6)]), 1)
        """
        n_modes = len(self.dom)
        if sum(x) != sum(y):
            raise ValueError("Number of photons in != Number of photons out.")
        matrix = np.stack([self.array[:, i] for i in range(n_modes)
                          for _ in range(y[i])], axis=1)
        matrix = np.stack([matrix[i] for i in range(n_modes)
                          for _ in range(x[i])], axis=0)
        divisor = np.prod([factorial(n) for n in y])
        return permanent(np.absolute(matrix)**2) / divisor

    def pdist_prob(self, x, y, S, permanent=npperm):
        """
        Calculates the probabilities for partially distinguishable photons.

        Parameters
        ----------
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers
        S : np.array
            Symmetric matrix of mutual distinguishabilities
            of shape (n_photons, n_photons)
        permanent : callable, optional
            Use another function for computing the permanent

        Example
        -------
        Check the Hong-Ou-Mandel effect:

        >>> BS = BBS(0)
        >>> x = [1, 1]
        >>> S = np.eye(2)
        >>> assert np.isclose(BS.pdist_prob(x, x, S), 0.5)
        >>> S = np.ones((2, 2))
        >>> assert np.isclose(BS.pdist_prob(x, x, S), 0)
        >>> S = lambda p: np.array([[1, p], [p, 1]])
        >>> for p in [0.1*x for x in range(11)]:
        ...     assert np.isclose(BS.pdist_prob(x, x, S(p)), 0.5 * (1 - p **2))
        """
        n_modes = len(self.dom)
        n_photons = sum(x)
        if sum(x) != sum(y):
            raise ValueError("Number of photons in != Number of photons out.")
        matrix = np.stack([self.array[:, i] for i in range(n_modes)
                          for _ in range(y[i])], axis=1)
        matrix = np.stack([matrix[i] for i in range(n_modes)
                          for _ in range(x[i])], axis=0)
        photons = list(range(n_photons))
        prob = 0
        for sigma in permutations(photons):
            for rho in permutations(photons):
                prob += np.prod([matrix[sigma[j], j]
                                 * np.conjugate(matrix[rho[j], j])
                                 * S[rho[j], sigma[j]] for j in photons])
        return prob

    def cl_distribution(self, x):
        """
        Computes the distribution of classical light in the outputs given
        an input distribution x.

        Parameters
        ----------
        x : List[float]
            Input vector of positive reals (intensities), expected to sum to 1.
            If the vector is not normalised the output will have the same
            normalisation factor.

        Example
        -------
        >>> TBS(0.25).cl_distribution([2/3, 1/3])
        array([0.5, 0.5])
        >>> assert np.allclose(TBS(0.25).cl_distribution([2/3, 1/3]),
        ...                    TBS(0.25).cl_distribution([1/5, 4/5]))
        >>> BS = TBS(0.25)
        >>> d = BS @ BS >> Id(1) @ BS @ Id(1)
        >>> d.cl_distribution([0, 1/2, 1/2, 0])
        array([0.25, 0.25, 0.25, 0.25])
        """
        return np.matmul(np.absolute(self.array)**2, np.array(x))

    def indist_prob_ub(self, x, y):
        """
        Evaluates probability of an optics.Diagram for input x and output y,
        when sending distinguishable particles,
        excluding bunching output probabilities.

        Parameters
        ----------
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers

        Example
        -------
        >>> BS = TBS(0.25)
        >>> d = BS @ BS >> Id(1) @ BS @ Id(1)
        >>> unbunch = [s for s in occupation_numbers(2, 4) if set(s)=={0,1}]
        >>> assert np.isclose(sum([d.indist_prob_ub([1, 1, 0, 0], y)
        ...                        for y in unbunch]), 1)
        """

        states = occupation_numbers(sum(x), len(x))
        bunch_states = [i for i in states if set(i) != {0, 1}]
        p_total = [self.indist_prob(x, i) for i in bunch_states]
        return self.indist_prob(x, y) / (1 - sum(p_total))


class Box(Diagram, monoidal.Box):
    """
    Box in an :py:class:`.optics.Diagram`.
    """
    def __init__(self, name, dom, cod, **params):
        if not isinstance(dom, PRO):
            raise TypeError(messages.type_err(PRO, dom))
        if not isinstance(cod, PRO):
            raise TypeError(messages.type_err(PRO, cod))
        monoidal.Box.__init__(self, name, dom, cod, **params)
        Diagram.__init__(self, dom, cod, [self], [0], layers=self.layers)

    def __repr__(self):
        return super().__repr__().replace('Box', 'optics.Box')


class PathBox(Box):
    """
    Box in Path category, see https://arxiv.org/abs/2204.12985.
    """
    def __repr__(self):
        return super().__repr__().replace('Box', 'PathBox')


class Monoid(PathBox):
    """Monoid :py:class:`~PathBox` in the Path category.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}1 \\\\ 1\\end{pmatrix}`.

    Also available under alias :py:obj:`monoid`.
    For more details see https://arxiv.org/abs/2204.12985."""
    def __init__(self):
        super().__init__('Monoid', PRO(2), PRO(1))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.shape = 'triangle_down'
        self.color = 'black'
        self.__class__.__repr__ = lambda _: 'monoid'

    def dagger(self):
        return comonoid

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [1, 1])


class Comonoid(PathBox):
    """Comonoid :py:class:`~PathBox` in the Path category.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}1 & 1\\end{pmatrix}`.

    Also available under alias :py:obj:`comonoid`.
    For more details see https://arxiv.org/abs/2204.12985."""
    def __init__(self):
        super().__init__('Comonoid', PRO(1), PRO(2))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.shape = 'triangle_up'
        self.color = 'black'
        self.__class__.__repr__ = lambda _: 'comonoid'

    def dagger(self):
        return monoid

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [1, 1])


class Unit(PathBox):
    """Unit :py:class:`~PathBox` in the Path category.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}\\end{pmatrix}`.

    Also available under alias :py:obj:`unit`.
    For more details see https://arxiv.org/abs/2204.12985."""
    def __init__(self):
        super().__init__('Unit', PRO(0), PRO(1))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'red'
        self.__class__.__repr__ = lambda _: 'unit'

    def dagger(self):
        return counit

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [])


class Counit(PathBox):
    """Counit :py:class:`~PathBox` in the Path category.

    Corresponds to :py:class:`.Matrix`
    :math:`\\begin{pmatrix}\\end{pmatrix}`.

    Also available under alias :py:obj:`counit`.
    For more details see https://arxiv.org/abs/2204.12985."""
    def __init__(self):
        super().__init__('Counit', PRO(1), PRO(0))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'red'
        self.__class__.__repr__ = lambda _: 'counit'

    def dagger(self):
        return unit

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [])


class Create(PathBox):
    """Creation :py:class:`~PathBox` in the QPath category.

    Also available under alias :py:obj:`create`.
    For more details see https://arxiv.org/abs/2204.12985."""
    def __init__(self):
        super().__init__('Create', PRO(0), PRO(1))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'black'
        self.__class__.__repr__ = lambda _: 'create'

    def dagger(self):
        return annil

    @property
    def matrix(self):
        raise Exception('Create has no Matrix semantics.')


class Annil(PathBox):
    """Annilation :py:class:`~PathBox` in the QPath category.

    Also available under alias :py:obj:`annil`.
    For more details see https://arxiv.org/abs/2204.12985."""
    def __init__(self):
        super().__init__('Annil', PRO(1), PRO(0))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'black'
        self.__class__.__repr__ = lambda _: 'annil'

    def dagger(self):
        return create

    @property
    def matrix(self):
        raise Exception('Annil has no Matrix semantics.')


class Scalar(PathBox):
    """ Scalar in QPath """
    def __init__(self, data):
        super().__init__("scalar", PRO(0), PRO(0), data=data)
        self.drawing_name = format_number(data)

    def dagger(self):
        return Scalar(self.data.conjugate())

    @property
    def matrix(self):
        raise Exception('Scalar has no Matrix semantics.')


class Endo(PathBox):
    """Endomorphism :py:class:`~PathBox` in the Path category.

    Corresponds to :py:class:`.Matrix`:
    :math:`\\begin{pmatrix} scalar \\end{pmatrix}`.

    For more details see https://arxiv.org/abs/2204.12985."""
    def __init__(self, scalar):
        super().__init__('Endo({})'.format(scalar), PRO(1), PRO(1))
        self.scalar = scalar
        try:
            self.drawing_name = f'{scalar:.3f}'
        except Exception:
            self.drawing_name = str(scalar)
        self.draw_as_spider = True
        self.shape = 'rectangle'
        self.color = 'green'
        self.__class__.__repr__ = lambda self: f'Endo({self.scalar})'

    @property
    def name(self):
        return f'Endo({self.scalar})'

    def dagger(self):
        return Endo(self.scalar.conjugate())

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [self.scalar])


#: Alias for :py:class:`Monoid() <discopy.quantum.optics.Monoid>`.
monoid = Monoid()
#: Alias for :py:class:`Monoid() <discopy.quantum.optics.Comonoid>`.
comonoid = Comonoid()
#: Alias for :py:class:`Unit() <discopy.quantum.optics.Unit>`.
unit = Unit()
#: Alias for :py:class:`Counit() <discopy.quantum.optics.Counit>`.
counit = Counit()
#: Alias for :py:class:`Create() <discopy.quantum.optics.Create>`.
create = Create()
#: Alias for :py:class:`Annil() <discopy.quantum.optics.Annil>`.
annil = Annil()


class Id(monoidal.Id, Diagram):
    """
    Identity for :py:class:`.optics.Diagram`.
    """
    def __init__(self, dom=PRO()):
        if isinstance(dom, int):
            dom = PRO(dom)
        monoidal.Id.__init__(self, dom)
        Diagram.__init__(self, dom, dom, [], [], layers=cat.Id(dom))


Diagram.id = Id


class Phase(Box):
    """
    Phase shifter.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix} e^{2\\pi i \\phi} \\end{pmatrix}`.

    Parameters
    ----------
    phi : float
        Phase parameter ranging from 0 to 1.

    Example
    -------
    >>> Phase(0.8).array
    array([[0.30901699-0.95105652j]])
    >>> assert np.allclose((Phase(0.4) >> Phase(0.4).dagger()).array
    ...                    , Id(1).array)
    """
    def __init__(self, phi):
        self.phi = phi
        super().__init__('Phase({})'.format({phi}), PRO(1), PRO(1))

    @property
    def matrix(self):
        import sympy
        backend = sympy if hasattr(self.phi, 'free_symbols') else np
        array = backend.exp(1j * 2 * backend.pi * self.phi)
        return Matrix(self.dom, self.cod, array)

    def dagger(self):
        return Phase(-self.phi)


class BBS(Box):
    """
    Beam splitter with a bias.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    \\tt{sin}((0.25 + bias)\\pi)
    & i \\tt{cos}((0.25 + bias)\\pi) \\\\
    i \\tt{cos}((0.25 + bias)\\pi)
    & \\tt{sin}((0.25 + bias)\\pi) \\end{pmatrix}`.

    Parameters
    ----------
    bias : float
        Bias from standard 50/50 beam splitter, parameter between 0 and 1.

    Example
    -------
    The standard beam splitter is:

    >>> BS = BBS(0)

    We can check the Hong-Ou-Mandel effect:

    >>> assert np.isclose(np.absolute(BS.amp([1, 1], [0, 2])) **2, 0.5)
    >>> assert np.isclose(np.absolute(BS.amp([1, 1], [2, 0])) **2, 0.5)
    >>> assert np.isclose(np.absolute(BS.amp([1, 1], [1, 1])) **2, 0)

    Check the dagger:

    >>> y = BBS(0.4)
    >>> assert np.allclose((y >> y.dagger()).eval(2), Id(2).eval(2))
    >>> comp = (y @ y >> Id(1) @ y @ Id(1)) >> (y @ y >> Id(1) @ y @ Id(1)
    ...   ).dagger()
    >>> assert np.allclose(comp.eval(2), Id(4).eval(2))
    """
    def __init__(self, bias):
        self.bias = bias
        super().__init__('BBS({})'.format(bias), PRO(2), PRO(2))

    def __repr__(self):
        return 'BS' if self.bias == 0 else super().__repr__()

    @property
    def matrix(self):
        sin = np.sin((0.25 + self.bias) * np.pi)
        cos = np.cos((0.25 + self.bias) * np.pi)
        array = [sin, 1j * cos, 1j * cos, sin]
        return Matrix(self.dom, self.cod, array)

    def dagger(self):
        return BBS(0.5 - self.bias)


class TBS(Box):
    """
    Tunable Beam Splitter.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    \\tt{sin}(\\theta \\, \\pi)
    & \\tt{cos}(\\theta \\, \\pi) \\\\
    \\tt{cos}(\\theta \\, \\pi) & - \\tt{sin}(\\theta \\, \\pi)
    \\end{pmatrix}`.

    Parameters
    ----------
    theta : float
        TBS parameter ranging from 0 to 1.

    Example
    -------
    >>> BS = BBS(0)
    >>> tbs = lambda x: BS >> Phase(x) @ Id(1) >> BS
    >>> assert np.allclose(TBS(0.15).array * TBS(0.15).global_phase,
    ...                    tbs(0.15).array)
    >>> assert np.allclose((TBS(0.25) >> TBS(0.25).dagger()).array,
    ...                    Id(2).array)
    >>> assert (TBS(0.25).dagger().global_phase ==
    ...         np.conjugate(TBS(0.25).global_phase))
    """
    def __init__(self, theta, _dagger=False):
        self.theta, self._dagger = theta, _dagger
        name = 'TBS({})'.format(theta)
        super().__init__(name, PRO(2), PRO(2), _dagger=_dagger)

    @property
    def global_phase(self):
        if self._dagger:
            return - 1j * np.exp(- 1j * self.theta * np.pi)
        else:
            return 1j * np.exp(1j * self.theta * np.pi)

    @property
    def matrix(self):
        sin = np.sin(self.theta * np.pi)
        cos = np.cos(self.theta * np.pi)
        array = [sin, cos, cos, -sin]
        return Matrix(self.dom, self.cod, array)

    def dagger(self):
        return TBS(self.theta, _dagger=True)


class MZI(Box):
    """
    Mach-Zender interferometer.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    e^{2\\pi i \\phi} \\tt{sin}(\\theta \\, \\pi)
    & \\tt{cos}(\\theta \\, \\pi) \\\\
    e^{2\\pi i \\phi} \\tt{cos}(\\theta \\, \\pi)
    & - \\tt{sin}(\\theta \\, \\pi) \\end{pmatrix}`.

    Parameters
    ----------
    theta: float
        Internal phase parameter, ranging from 0 to 1.
    phi: float
        External phase parameter, ranging from 0 to 1.

    Example
    -------
    >>> assert np.allclose(MZI(0.28, 0).array, TBS(0.28).array)
    >>> assert np.isclose(MZI(0.28, 0.3).global_phase, TBS(0.28).global_phase)
    >>> assert np.isclose(MZI(0.12, 0.3).global_phase.conjugate(),
    ...                   MZI(0.12, 0.3).dagger().global_phase)
    >>> mach = lambda x, y: TBS(x) >> Phase(y) @ Id(1)
    >>> assert np.allclose(MZI(0.28, 0.9).array, mach(0.28, 0.9).array)
    >>> assert np.allclose((MZI(0.28, 0.34) >> MZI(0.28, 0.34).dagger()).array,
    ...                    Id(2).array)
    """
    def __init__(self, theta, phi, _dagger=False):
        self.theta, self.phi, self._dagger = theta, phi, _dagger
        super().__init__('MZI', PRO(2), PRO(2), _dagger=_dagger)

    @property
    def global_phase(self):
        if not self._dagger:
            return 1j * np.exp(1j * self.theta * np.pi)
        else:
            return - 1j * np.exp(- 1j * self.theta * np.pi)

    @property
    def matrix(self):
        import sympy
        backend = sympy if hasattr(self.theta, 'free_symbols') else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        exp = backend.exp(1j * 2 * backend.pi * self.phi)
        array = np.array([exp * sin, cos, exp * cos, -sin])
        matrix = Matrix(self.dom, self.cod, array)
        matrix = matrix.dagger() if self._dagger else matrix
        return matrix

    def dagger(self):
        return MZI(self.theta, self.phi, _dagger=not self._dagger)


class Functor(monoidal.Functor):
    """ Can be used for catching lions """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=PRO, ar_factory=Diagram)


def params_shape(width, depth):
    """
    Returns shape of parameters for :py:func:`ansatz` given width and depth.
    """
    even_width = not width % 2
    even_depth = not depth % 2
    if even_width:
        if even_depth:
            # we have width // 2 MZIs on the first row
            # followed by width // 2 - 1 equals width - 1
            return (depth // 2, width - 1, 2)
        else:
            # we have the parameters for even depths plus
            # a last layer of width // 2 MZIs
            return (depth // 2 * (width - 1) + width // 2, 2)
    else:
        # we have width // 2 MZIs on each row, where
        # the even layers are tensored by Id on the right
        # and the odd layers are tensored on the left.
        return (depth, width // 2, 2)


def ansatz(width, depth, x):
    """
    Returns a universal interferometer given width, depth and parameters x,
    based on https://arxiv.org/abs/1603.08788.
    """
    params = x.reshape(params_shape(width, depth))
    chip = Id(width)
    if not width % 2:
        if depth % 2:
            params, last_layer = params[:-width // 2].reshape(
                params_shape(width, depth - 1)), params[-width // 2:]
        for i in range(depth // 2):
            chip = chip\
                >> Id().tensor(*[
                    MZI(*params[i, j])
                    for j in range(width // 2)])\
                >> Id(1) @ Id().tensor(*[
                    MZI(*params[i, j + width // 2])
                    for j in range(width // 2 - 1)]) @ Id(1)
        if depth % 2:
            chip = chip >> Id().tensor(*[
                MZI(*last_layer[j]) for j in range(width // 2)])
    else:
        for i in range(depth):
            left, right = (Id(1), Id()) if i % 2 else (Id(), Id(1))
            chip >>= left.tensor(*[
                MZI(*params[i, j])
                for j in range(width // 2)]) @ right
    return chip


#: Alias for :py:class:`BBS(0) <discopy.quantum.optics.BBS>`.
BS = BBS(0)


def to_matrix(diagram):
    return monoidal.Functor(
        ob=lambda x: x, ar=lambda x: x.matrix,
        ob_factory=PRO, ar_factory=Matrix)(diagram)


def ar_optics2path(box):
    if isinstance(box, Phase):
        import sympy
        backend = sympy if hasattr(box.phi, 'free_symbols') else np
        return Endo(backend.exp(2j * backend.pi * box.phi))
    if isinstance(box, TBS):
        sin = np.sin(box.theta * np.pi)
        cos = np.cos(box.theta * np.pi)
        array = Id().tensor(*map(Endo, (sin, cos, cos, -sin)))
        w1, w2 = comonoid, monoid
        return w1 @ w1 >> array.permute(2, 1) >> w2 @ w2
    if isinstance(box, BBS):
        sin = np.sin((0.25 + box.bias) * np.pi)
        cos = np.cos((0.25 + box.bias) * np.pi)
        array = Id().tensor(*map(Endo, (sin, 1j * cos, 1j * cos, sin)))
        w1, w2 = comonoid, monoid
        return w1 @ w1 >> array.permute(2, 1) >> w2 @ w2
    if isinstance(box, MZI):
        phi, theta = box.phi, box.theta
        cos = np.cos(np.pi * theta)
        sin = np.sin(np.pi * theta)
        exp = np.exp(1j * 2 * np.pi * phi)
        array = Id().tensor(*map(Endo, (exp * sin, cos, exp * cos, -sin)))
        w1, w2 = comonoid, monoid
        return w1 @ w1 >> array.permute(2, 1) >> w2 @ w2
    raise NotImplementedError


optics2path = Functor(ob=lambda x: x, ar=ar_optics2path)


def ar_zx2path(box):
    from discopy.quantum import zx
    n, m = len(box.dom), len(box.cod)
    if isinstance(box, zx.Scalar):
        return Scalar(box.data)
    if isinstance(box, zx.X):
        phase = box.phase
        root2 = Scalar(2 ** 0.5)
        if (n, m, phase) == (0, 1, 0):
            return create @ unit @ root2
        if (n, m, phase) == (0, 1, 0.5):
            return unit @ create @ root2
        if (n, m, phase) == (1, 0, 0):
            return annil @ counit @ root2
        if (n, m, phase) == (1, 0, 0.5):
            return counit @ annil @ root2
        if (n, m, phase) == (1, 1, 0.25):
            return BBS(0.5)  # = BS.dagger()
        if (n, m, phase) == (1, 1, -0.25):
            return BS
    if isinstance(box, zx.Z):
        phase = box.phase
        if (n, m, phase) == (0, 2, 0):
            plus = create >> comonoid
            fusion = plus >> Id(1) @ plus @ Id(1)
            d = (fusion @ fusion
                 >> Id(2) @ BBS(0.5) @ BS @ Id(2)
                 >> Id(2) @ fusion.dagger() @ Id(2))
            return d
        if (n, m) == (0, 1):
            return create >> comonoid
        if (n, m) == (1, 1):
            return Phase(-phase / 2) @ Phase(phase / 2)
        if (n, m, phase) == (2, 1, 0):
            return Id(1) @ (Monoid() >> annil) @ Id(1)
        if (n, m, phase) == (1, 2, 0):
            plus = create >> comonoid
            bot = (plus >> Id(1) @ plus @ Id(1)) @ (Id(1) @ plus @ Id(1))
            mid = Id(2) @ BS.dagger() @ BS @ Id(2)
            fusion = Id(1) @ plus.dagger() @ Id(1) >> plus.dagger()
            return bot >> mid >> (Id(2) @ fusion @ Id(2))
    if box == zx.H:
        return TBS(0.25)
    raise NotImplementedError(f'No translation of {box} in QPath.')


zx2path = Functor(ob=lambda x: x @ x, ar=ar_zx2path)


def swap_right(diagram, i):
    left, box, right = diagram.layers[i]
    if box.dom:
        raise ValueError(f"{box} is not a state.")

    new_left, new_right = left @ right[0:1], right[1:]
    new_layer = diagram.id(new_left) @ box @ diagram.id(new_right)
    return (
        diagram[:i]
        >> new_layer.permute(len(new_left), len(new_left) - 1)
        >> diagram[i + 1:])


def drag_out(diagram, i):
    box = diagram.boxes[i]
    if box.dom:
        raise ValueError(f"{box} is not a state.")
    while i > 0:
        try:
            diagram = diagram.interchange(i - 1, i)
            i -= 1
        except InterchangerError:
            diagram = swap_right(diagram, i)
    return diagram


def drag_all(diagram):
    i = len(diagram) - 1
    stop = 0
    while i >= stop:
        box = diagram.boxes[i]
        if box == create or box == unit:
            diagram = drag_out(diagram, i)
            i = len(diagram) - 1
            stop += 1
        i -= 1
    return diagram


def remove_scalars(diagram):
    new_diagram = diagram
    scalar, num = 1, 0
    for i, box in enumerate(diagram.boxes):
        if isinstance(box, Scalar):
            new_diagram = new_diagram[:i - num] >> new_diagram[i + 1 - num:]
            num += 1
            scalar *= box.data
    return new_diagram, scalar


def qpath_drag(diagram):
    """
    Drag :py:class:`.Create`s, :py:class:`.Annil`s, :py:class:`.Unit`s and
    :py:class:`.Counit`s to the top and bottom of the diagram.
    """
    diagram, scalar = remove_scalars(diagram)
    diagram = drag_all(diagram)
    diagram = drag_all(diagram.dagger()).dagger()
    n_state = len([b for b in diagram.boxes if b in (create, unit)])
    n_costate = len([b for b in diagram.boxes if b in (annil, counit)])
    top, bot = diagram[:n_state], diagram[len(diagram) - n_costate:]
    mat = diagram[n_state:len(diagram) - n_costate]
    return top, bot, mat, scalar


def evaluate(diagram, inp, out):
    """ Evaluate the amplitude of <J|Diagram|I>. """
    assert len(inp) == len(diagram.dom) and len(out) == len(diagram.cod)
    top, bot, drag, scalar = qpath_drag(diagram)
    inp, out = inp[:], out[:]
    x = top.normal_form()
    y = bot.dagger().normal_form()
    for box, off in zip(x.boxes, x.offsets):
        if box == create:
            inp.insert(off, 1)
        if box == unit:
            inp.insert(off, 0)
    for box, off in zip(y.boxes, y.offsets):
        if box == create:
            out.insert(off, 1)
        if box == unit:
            out.insert(off, 0)
    if sum(inp) != sum(out):
        raise ValueError('# of photons in != # of photons out')
    matrix = to_matrix(drag).array
    n_modes_in = len(drag.dom)
    n_modes_out = len(drag.cod)
    matrix = np.stack([matrix[:, i] for i in range(n_modes_out)
                      for _ in range(out[i])], axis=1)
    matrix = np.stack([matrix[i] for i in range(n_modes_in)
                      for _ in range(inp[i])], axis=0)
    divisor = np.sqrt(np.prod([factorial(n) for n in inp + out]))
    return scalar * npperm(matrix) / divisor


def ar_make_square(box):
    comon = unit @ Id(1) >> BS >> Endo(-1j) @ Id(1) @ Scalar(2 ** .5)
    mon = comon.dagger()
    if box == monoid:
        return mon
    if box == comonoid:
        return comon
    return box


make_square = Functor(ob=lambda x: x, ar=ar_make_square)
