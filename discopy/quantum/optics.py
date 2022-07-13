# -*- coding: utf-8 -*-

"""
Implements linear optics
"""

from cmath import phase
import numpy as np
from math import factorial, sqrt
from itertools import permutations

from discopy import cat, monoidal
from discopy.monoidal import PRO

from discopy.matrix import Matrix
import sympy


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
    """
    def __repr__(self):
        return super().__repr__().replace('Diagram', 'optics.Diagram')

    @property
    def array(self):
        """
        The array corresponding to the diagram.
        Builds a block diagonal matrix for each layer and then multiplies them
        in sequence.

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
        when sending INDISTINGUISHABLE photons.

        Parameters
        ----------
        x : List[int]
            Input vector of occupation numbers
        y : List[int]
            Output vector of occupation numbers
        permanent : callable, optional
            Use another function for computing the permanent
            or set permanent = np.determinant to compute fermionic statistics

        >>> network = MZI(0.2, 0.4) @ MZI(0.2, 0.4)\
                      >> Id(1) @ MZI(0.2, 0.4) @ Id(1)
        >>> amplitude = network.amp([1, 0, 0, 1], [1, 0, 1, 0])
        >>> probability = np.abs(amplitude) ** 2
        >>> assert probability > 0.05
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
        Evaluates the matrix acting on the Fock space given number of photons.

        Parameters
        ----------
        n_photons : int
            Number of photons
        permanent : callable, optional
            Use another function for computing the permanent
            (e.g. from thewalrus)

        >>> for i, _ in enumerate(occupation_numbers(3, 2)): assert np.isclose(
        ...       sum(np.absolute(MZI(0.2, 0.4).eval(3)[i])**2), 1)
        >>> network = MZI(0.2, 0.4) @ MZI(0.2, 0.4)\
                      >> Id(1) @ MZI(0.2, 0.4) @ Id(1)
        >>> for i, _ in enumerate(occupation_numbers(2, 4)): assert np.isclose(
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

        >>> box = MZI(1.2, 0.6)
        >>> assert np.isclose(sum([box.dist_prob([3, 0], y)
        ...                        for y in occupation_numbers(3, 2)]), 1)
        >>> network = box @ box @ box >> Id(1) @ box @ box @ Id(1)
        >>> assert np.isclose(sum([network.dist_prob([0, 1, 0, 1, 1, 1], y)
        ...                        for y in occupation_numbers(4, 6)]), 1)
        """
        n_modes = len(self.dom)
        if sum(x) != sum(y):
            return 0
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

        Check Hong-Ou-Mandel
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
            return 0
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
    Box in an optics.Diagram
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
    Box in Path category.
    """
    def __repr__(self):
        return super().__repr__().replace('Box', 'PathBox')


class Monoid(PathBox):
    """W spider"""
    def __init__(self):
        super().__init__('Monoid', PRO(2), PRO(1))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.shape = 'triangle_up'
        self.color = 'black'

    def dagger(self):
        return Comonoid()

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [1, 1])


class Comonoid(PathBox):
    """W spider"""
    def __init__(self):
        super().__init__('Comonoid', PRO(1), PRO(2))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.shape = 'triangle_down'
        self.color = 'black'

    def dagger(self):
        return Monoid()

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [1, 1])


class Unit(PathBox):
    """Red node"""
    def __init__(self):
        super().__init__('Unit', PRO(0), PRO(1))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'red'

    def dagger(self):
        return Counit()

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [])


class Counit(PathBox):
    """Red node"""
    def __init__(self):
        super().__init__('Unit', PRO(1), PRO(0))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'red'

    def dagger(self):
        return Counit()

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [])


class Create(PathBox):
    """Black node"""
    def __init__(self):
        super().__init__('Create', PRO(0), PRO(1))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'black'

    def dagger(self):
        return Annil()

    @property
    def matrix(self):
        raise Exception('Create has no Matrix semantics.')


class Annil(PathBox):
    """Black node"""
    def __init__(self):
        super().__init__('Annil', PRO(1), PRO(0))
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'black'

    def dagger(self):
        return Create()

    @property
    def matrix(self):
        raise Exception('Annil has no Matrix semantics.')


class Endo(PathBox):
    """Green box"""
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

    @property
    def name(self):
        return f'Endo({self.scalar})'

    def dagger(self):
        return Endo(phase.conjugate())

    @property
    def matrix(self):
        return Matrix(self.dom, self.cod, [self.scalar])


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


class Phase(Box):
    """
    Phase shifter.

    Parameters
    ----------
    phi : float
    Phase parameter ranging from 0 to 1.

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
        backend = sympy if hasattr(self.phi, 'free_symbols') else np
        array = backend.exp(1j * 2 * backend.pi * self.phi)
        return Matrix(self.dom, self.cod, array)

    def dagger(self):
        return Phase(-self.phi)


class BBS(Box):
    """
    Beam splitter with a bias.

    Parameters
    ----------
    bias : float
    Bias angle from standard 50/50 beam splitter, parameter between 0 and 1.

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

    Parameters
    ----------
    theta : float
    Beam splitter parameter ranging from 0 to 1.

    >>> BS = BBS(0)
    >>> tbs = lambda x: BS >> Phase(x) @ Id(1) >> BS
    >>> assert np.allclose(TBS(0.15).array * TBS(0.15).global_phase,
    ...                    tbs(0.15).array)
    >>> assert np.allclose((TBS(0.25) >> TBS(0.25).dagger()).array, Id(2).array)
    >>> assert TBS(0.25).dagger().global_phase == np.conjugate(TBS(0.25).global_phase)
    """
    def __init__(self, theta, _dagger=False):
        self.theta, self._dagger = theta, _dagger
        super().__init__('TBS({})'.format(theta), PRO(2), PRO(2), _dagger=_dagger)

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
    Mach-Zender interferometer

    Parameters
    ----------
    theta, phi : float
    Internal and external phase parameters of the MZI, ranging from 0 to 1.

    >>> assert np.allclose(MZI(0.28, 0).array, TBS(0.28).array)
    >>> assert np.isclose(MZI(0.28, 0.3).global_phase, TBS(0.28).global_phase)
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
        backend = sympy if hasattr(self.theta, 'free_symbols') else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        if self._dagger:
            exp = backend.exp(- 1j * 2 * backend.pi * self.phi)
            array = np.array([exp * sin, exp * cos, cos, -sin])
            return Matrix(self.dom, self.cod, array)
        else:
            exp = backend.exp(1j * 2 * backend.pi * self.phi)
            array = np.array([exp * sin, cos, exp * cos, -sin])
            return Matrix(self.dom, self.cod, array)

    def dagger(self):
        return MZI(self.theta, self.phi, _dagger=not self._dagger)


class Functor(monoidal.Functor):
    """ Can be used for catching lions """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=PRO, ar_factory=Diagram)


def params_shape(width, depth):
    """ Returns the shape of parameters given width and depth. """
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
    """ Returns an array of MZIs given width, depth and parameters x"""
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


BS = BBS(0)


def to_matrix(diagram):
    return monoidal.Functor(
        ob=lambda x: x, ar=lambda x: x.matrix,
        ob_factory=PRO, ar_factory=Matrix)(diagram)


def ar_to_path(box):
    if isinstance(box, Phase):
        backend = sympy if hasattr(box.phi, 'free_symbols') else np
        return Endo(backend.exp(2j * backend.pi * box.phi))
    if isinstance(box, TBS):
        r, t = box.r, box.t
        array = Id().tensor(*map(Endo, (r, t, t.conjugate(), -r.conjugate())))
        w1, w2 = Comonoid(), Monoid()
        return w1 @ w1 >> array.permute(2, 1) >> w2 @ w2
    if isinstance(box, MZI):
        phi, theta = box.phi, box.theta
        diagram = (
            beam_splitter >> Id(PRO(1)) @ Phase(phi) >>
            beam_splitter >> Id(PRO(1)) @ Phase(theta))
        return to_path(diagram)


to_path = Functor(ob=lambda x: x, ar=ar_to_path)


def qpath_drag(diagram):
    done = False
    j = 0
    while not done:
        for i in range(j, len(diagram)):
            box = diagram.boxes[i]
            if box.name == 'Create':
                while i > j:
                    diagram = diagram.interchange(i - 1, i)
                    i -= 1
                j += 1
                break
        done = i == len(diagram) - 1
    done = False
    j = len(diagram) - 1
    while not done:
        for i in range(j, 0, -1):
            box = diagram.boxes[i]
            if box.name == 'Annil':
                while i < j:
                    diagram = diagram.interchange(i, i + 1)
                    i += 1
                j -= 1
                break
        done = i == 1
    return diagram
 