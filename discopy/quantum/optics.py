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

from discopy.quantum.oplus import Matrix
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

    >>> grid = MZI(0.5, 0.3) @ MZI(0.5, 0.3) >> Id(1) @ MZI(0.5, 0.3) @ Id(1)
    >>> assert np.allclose((grid >> grid.dagger()).eval(3), Id(4).eval(3))
    """
    def __repr__(self):
        return super().__repr__().replace('Diagram', 'optics.Diagram')

    @property
    def array(self):
        """
        The array corresponding to the diagram.
        Builds a block diagonal matrix for each layer and then multiplies them
        in sequence.

        >>> BS = BeamSplitter(0.5)
        >>> np.shape(to_matrix(BS).array)
        (2, 2)
        >>> np.shape(to_matrix(BS >> BS).array)
        (2, 2)
        >>> np.shape(to_matrix(BS @ BS @ BS).array)
        (6, 6)
        >>> assert np.allclose(
        ...     to_matrix(MZI(0, 0)).array, np.array([[0, 1], [1, 0]])
        >>> assert np.allclose(
        ...     to_matrix(MZI(0, 0) >> MZI(0, 0)).array, to_matrix(Id(2)).array)
        """
        return to_matrix(self)

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
        >>> amplitude
        (-0.08637287570313157-0.26582837761001243j)
        >>> probability = np.abs(amplitude) ** 2
        >>> probability
        0.07812499999999997
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

        >>> box = MZI(1.2, 0.6)
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
        >>> BS = BeamSplitter(0.5)
        >>> x = [1, 1]
        >>> S = np.eye(2)
        >>> assert np.isclose(BS.pdist_prob(x, x, S), 0.5)
        >>> S = np.ones((2, 2))
        >>> assert np.isclose(BS.pdist_prob(x, x, S), 0.0)
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

        >>> BeamSplitter(0.5).cl_distribution([2/3, 1/3])
        array([0.5, 0.5])
        >>> assert np.allclose(BeamSplitter(0.5).cl_distribution([2/3, 1/3]),
        ...                    BeamSplitter(0.5).cl_distribution([1/5, 4/5]))
        >>> BS = BeamSplitter(0.25)
        >>> d = BS @ BS >> Id(1) @ BS @ Id(1)
        >>> d.cl_distribution([0, 1/2, 1/2, 0])
        array([0.4267767, 0.0732233, 0.0732233, 0.4267767])
        """
        return np.matmul(np.absolute(self.array)**2, np.array(x))


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
        """ The array or unitary inside the box. """
        return Matrix(self.dom, self.cod, self.data)


class PathBox(Box):
    """
    Box in Path category.
    """
    def __repr__(self):
        return super().__repr__().replace('Box', 'PathBox')


class Monoid(PathBox):
    """W spider"""
    def __init__(self):
        super().__init__('Monoid', PRO(2), PRO(1), [])
        self.drawing_name = ''
        self.draw_as_spider = True
        self.shape = 'triangle_up'
        self.color = 'black'

    def dagger(self):
        return Comonoid()

    @property
    def array(self):
        return Matrix(self.dom, self.cod, [1, 1])


class Comonoid(PathBox):
    """W spider"""
    def __init__(self):
        super().__init__('Comonoid', PRO(1), PRO(2), [])
        self.drawing_name = ''
        self.draw_as_spider = True
        self.shape = 'triangle_down'
        self.color = 'black'

    def dagger(self):
        return Monoid()

    @property
    def array(self):
        return Matrix(self.dom, self.cod, [1, 1])


class Unit(PathBox):
    """Red node"""
    def __init__(self):
        super().__init__('Unit', PRO(0), PRO(1), [])
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'red'

    def dagger(self):
        return Counit()

    @property
    def array(self):
        return Matrix(self.dom, self.cod, [])


class Counit(PathBox):
    """Red node"""
    def __init__(self):
        super().__init__('Unit', PRO(1), PRO(0), [])
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'red'

    def dagger(self):
        return Counit()

    @property
    def array(self):
        return Matrix(self.dom, self.cod, [])


class Create(PathBox):
    """Black node"""
    def __init__(self):
        super().__init__('Create', PRO(0), PRO(1), [])
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'black'

    def dagger(self):
        return Annil()

    @property
    def array(self):
        raise Exception('Create has no Matrix semantics.')


class Annil(PathBox):
    """Black node"""
    def __init__(self):
        super().__init__('Annil', PRO(1), PRO(0), [])
        self.drawing_name = ''
        self.draw_as_spider = True
        self.color = 'black'

    def dagger(self):
        return Create()

    @property
    def array(self):
        raise Exception('Annil has no Matrix semantics.')


class Endo(PathBox):
    """Green box"""
    def __init__(self, scalar):
        super().__init__('Endo', PRO(1), PRO(1), scalar)
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
    def array(self):
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


class PhaseShift(Box):
    """
    Phase shifter

    Parameters
    ----------
    phase : float

    >>> PhaseShift(0.4).array
    array(-0.80901699+0.58778525j)
    >>> assert np.allclose((PhaseShift(0.4) >> PhaseShift(0.4).dagger()).array
    ...                    , Id(1).array)
    """
    def __init__(self, phase):
        self.phase = phase
        super().__init__('Phase shift', PRO(1), PRO(1), phase)

    @property
    def array(self):
        backend = sympy if hasattr(self.phase, 'free_symbols') else np
        array = backend.exp(2j * backend.pi * self.phase)
        return Matrix(self.dom, self.cod, array)

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

    We can check the Hong-Ou-Mandel effect:
    >>> BS = BeamSplitter(0.5)
    >>> assert np.isclose(np.absolute(BS.amp([1, 1], [0, 2])) **2, 0.5)
    >>> assert np.isclose(np.absolute(BS.amp([1, 1], [2, 0])) **2, 0.5)
    >>> assert np.isclose(np.absolute(BS.amp([1, 1], [1, 1])) **2, 0)
    """
    def __init__(self, r, t):
        self.r = r
        self.t = t
        super().__init__('Beam splitter', PRO(2), PRO(2), [r, t])

    @property
    def array(self):
        r, t = self.r, self.t
        array = [r, t, t.conjugate(), -r.conjugate()]
        return Matrix(self.dom, self.cod, array)

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
    >>> assert np.allclose(MZI(0.4, 0.9).eval(4), mach(0.4, 2 * 0.9).eval(4))
    >>> assert np.allclose((MZI(0.4, 0.9) >> MZI(0.4, 0.9).dagger()).eval(3),
    ...                     Id(2).eval(3))
    """
    def __init__(self, phase, angle, _dagger=False):
        self.phase, self.angle, self._dagger = phase, angle, _dagger
        super().__init__('MZI', PRO(2), PRO(2), data=[phase, angle],
                         _dagger=_dagger)

    @property
    def array(self):
        backend = sympy if hasattr(self.angle, 'free_symbols') else np
        cos = backend.cos(backend.pi * self.angle)
        sin = backend.sin(backend.pi * self.angle)
        exp = backend.exp(2j * backend.pi * self.phase)
        array = np.array([exp * sin, exp * cos, cos, -sin])
        return Matrix(self.dom, self.cod, array)

    def dagger(self):
        return MZI(self.phase, self.angle, _dagger=not self._dagger)


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


beam_splitter = BeamSplitter(1j * 2 ** -0.5, 2 ** -0.5)


def to_matrix(diagram):
    return monoidal.Functor(
        ob=lambda x: x, ar=lambda x: x.array,
        ob_factory=PRO, ar_factory=Matrix)(diagram)


def ar_to_path(box):
    if isinstance(box, PhaseShift):
        backend = sympy if hasattr(box.phase, 'free_symbols') else np
        return Endo(backend.exp(2j * backend.pi * box.phase))
    if isinstance(box, BeamSplitter):
        r, t = box.r, box.t
        array = Id().tensor(*map(Endo, (r, t, t.conjugate(), -r.conjugate())))
        w1, w2 = Comonoid(), Monoid()
        return w1 @ w1 >> array.permute(2, 1) >> w2 @ w2
    if isinstance(box, MZI):
        phase, angle = box.phase, box.angle
        diagram = (
            beam_splitter >> Id(PRO(1)) @ PhaseShift(phase) >>
            beam_splitter >> Id(PRO(1)) @ PhaseShift(angle))
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
