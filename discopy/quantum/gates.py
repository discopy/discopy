# -*- coding: utf-8 -*-

""" Gates in a :class:`discopy.quantum.circuit.Circuit`. """

from collections.abc import Callable
from discopy.cat import AxiomError, recursive_subs
from discopy.tensor import np, Dim, Tensor
from discopy.quantum.circuit import bit, qubit, Box, Swap, Sum, Id


def format_number(data):
    """ Tries to format a number. """
    try:
        return '{:.3g}'.format(data)
    except TypeError:
        return data


class QuantumGate(Box):
    """ Quantum gates, i.e. unitaries on n qubits. """
    def __init__(self, name, n_qubits, array=None, data=None, _dagger=False):
        dom = qubit ** n_qubits
        if array is not None:
            self._array = np.array(array).reshape(
                2 * n_qubits * (2, ) or (1, ))
        super().__init__(
            name, dom, dom, is_mixed=False, data=data, _dagger=_dagger)

    @property
    def array(self):
        """ The array of a quantum gate. """
        return self._array

    def __repr__(self):
        if self in GATES:
            return self.name
        return "QuantumGate({}, n_qubits={}, array={})".format(
            repr(self.name), len(self.dom),
            np.array2string(self.array.flatten()))

    def dagger(self):
        return QuantumGate(
            self.name, len(self.dom), self.array,
            _dagger=None if self._dagger is None else not self._dagger)


class ClassicalGate(Box):
    """ Classical gates, i.e. from bits to bits. """
    def __init__(self, name, n_bits_in, n_bits_out, data=None, _dagger=False):
        dom, cod = bit ** n_bits_in, bit ** n_bits_out
        if isinstance(data, Callable):
            self.is_linear = False
        else:
            self.is_linear = True
            data = np.array(data).reshape(
                (n_bits_in + n_bits_out) * (2, ) or (1, ))
        super().__init__(
            name, dom, cod, is_mixed=False, data=data, _dagger=_dagger)

    @property
    def array(self):
        """ The array of a classical gate. """
        if self.is_linear:
            return self.data
        raise AttributeError("{} is non-linear.".format(self))

    @property
    def func(self):
        """ The underlying function of a classical gate. """
        if self.is_linear:
            return lambda other: other >> self.eval()

        def apply(state):
            dom, cod = Dim(*(len(self.dom) * [2])), Dim(*(len(self.cod) * [2]))
            if (state.dom, state.cod) != (Dim(1), dom):
                raise AxiomError("Non-linear gates can only be applied "
                                 "to states, not processes.")
            return Tensor(Dim(1), cod, self.data(state.array))
        return apply

    def __call__(self, other):
        return self.func(other)

    def __eq__(self, other):
        if not isinstance(other, ClassicalGate):
            return super().__eq__(other)
        return (self.name, self.dom, self.cod)\
            == (other.name, other.dom, other.cod)\
            and np.all(self.array == other.array)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        data = np.array2string(self.array.flatten())\
            if self.is_linear else self.data
        return "ClassicalGate({}, n_bits_in={}, n_bits_out={}, data={})"\
            .format(repr(self.name), len(self.dom), len(self.cod), data)

    def dagger(self):
        return ClassicalGate(
            self.name, len(self.cod), len(self.dom), self.array,
            _dagger=None if self._dagger is None else not self._dagger)

    def subs(self, *args):
        data = recursive_subs(self.data, *args)
        return ClassicalGate(self.name, len(self.cod), len(self.dom), data)

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        name = "{}.grad({})".format(self.name, var)
        n_bits_in, n_bits_out = len(self.dom), len(self.cod)
        array = self.eval().grad(var).array
        return ClassicalGate(name, n_bits_in, n_bits_out, array)


class Copy(ClassicalGate):
    """ Takes a bit, returns two copies of it. """
    def __init__(self):
        super().__init__("Copy", 1, 2, [1, 0, 0, 0, 0, 0, 0, 1])
        self.draw_as_spider, self.color = True, "black"
        self.drawing_name = ""

    def dagger(self):
        return Match()


class Match(ClassicalGate):
    """ Takes two bits in, returns them if they are equal. """
    def __init__(self):
        super().__init__("Match", 2, 1, [1, 0, 0, 0, 0, 0, 0, 1])
        self.draw_as_spider, self.color = True, "black"
        self.drawing_name = ""

    def dagger(self):
        return Copy()


class Bits(ClassicalGate):
    """
    Implements bit preparation for a given bitstring.

    >>> assert Bits(1, 0).cod == bit ** 2
    >>> assert Bits(1, 0).eval()\\
    ...     == Tensor(dom=Dim(1), cod=Dim(2, 2), array=[0, 0, 1, 0])
    """
    def __init__(self, *bitstring, _dagger=False):
        utensor = Tensor.id(Dim(1)).tensor(*(
            Tensor(Dim(1), Dim(2), [0, 1] if bit else [1, 0])
            for bit in bitstring))
        name = "Bits({})".format(', '.join(map(str, bitstring)))
        dom, cod = (len(bitstring), 0) if _dagger else (0, len(bitstring))
        super().__init__(name, dom, cod, data=utensor.array, _dagger=_dagger)
        self.bitstring = bitstring

    def __repr__(self):
        return self.name + (".dagger()" if self._dagger else "")

    def dagger(self):
        return Bits(*self.bitstring, _dagger=not self._dagger)


class Ket(Box):
    """
    Implements qubit preparation for a given bitstring.

    >>> assert Ket(1, 0).cod == qubit ** 2
    >>> assert Ket(1, 0).eval()\\
    ...     == Tensor(dom=Dim(1), cod=Dim(2, 2), array=[0, 0, 1, 0])
    """
    def __init__(self, *bitstring):
        dom, cod = qubit ** 0, qubit ** len(bitstring)
        name = "Ket({})".format(', '.join(map(str, bitstring)))
        super().__init__(name, dom, cod, is_mixed=False)
        self.bitstring = bitstring
        self.array = Bits(*bitstring).array

    def dagger(self):
        return Bra(*self.bitstring)


class Bra(Box):
    """
    Implements qubit post-selection for a given bitstring.

    >>> assert Bra(1, 0).dom == qubit ** 2
    >>> assert Bra(1, 0).eval()\\
    ...     == Tensor(dom=Dim(2, 2), cod=Dim(1), array=[0, 0, 1, 0])
    """
    def __init__(self, *bitstring):
        name = "Bra({})".format(', '.join(map(str, bitstring)))
        dom, cod = qubit ** len(bitstring), qubit ** 0
        super().__init__(name, dom, cod, is_mixed=False)
        self.bitstring = bitstring
        self.array = Bits(*bitstring).array

    def dagger(self):
        return Ket(*self.bitstring)


class Parametrized(Box):
    """
    Abstract class for parametrized boxes in a quantum circuit.

    Parameters
    ----------
    name : str
        Name of the parametrized class, e.g. :code:`"CRz"`.
    dom, cod : BitsAndQubits
        Domain and codomain.
    data : any
        Data of the box, potentially with free symbols.
    datatype : type
        Type to cast whenever there are no free symbols.

    Example
    -------
    >>> from sympy.abc import phi
    >>> from sympy import pi, exp, I
    >>> assert Rz(phi)\\
    ...     == Parametrized('Rz', qubit, qubit, data=phi, is_mixed=False)
    >>> assert Rz(phi).array[0,0] == exp(-1.0 * I * pi * phi)
    >>> assert list((Rz(phi) >> Rz(-phi)).eval()
    ...             .array.flatten()) == [1, 0, 0, 1]
    """
    def __init__(self, name, dom, cod, data=None, **params):
        self._datatype = params.get('datatype', None)
        data = data\
            if getattr(data, "free_symbols", False) else self._datatype(data)
        self.drawing_name = '{}({})'.format(name, data)
        Box.__init__(
            self, name, dom, cod, data=data,
            is_mixed=params.get('is_mixed', True),
            _dagger=params.get('_dagger', False))
        if self.free_symbols:
            import sympy
            self._pi, self._exp = sympy.pi, sympy.exp
            self._cos, self._sin = sympy.cos, sympy.sin
        else:
            self._pi, self._exp = np.pi, np.exp
            self._cos, self._sin = np.cos, np.sin

    def subs(self, *args):
        data = recursive_subs(self.data, *args)
        return type(self)(data)

    @property
    def name(self):
        return '{}({})'.format(self._name, format_number(self.data))

    def __repr__(self):
        return self.name


class Rotation(Parametrized, QuantumGate):
    """ Abstract class for rotation gates. """
    def __init__(self, phase, name=None, n_qubits=1):
        QuantumGate.__init__(self, name, n_qubits)
        Parametrized.__init__(
            self, name, self.dom, self.cod,
            datatype=float, is_mixed=False, data=phase)

    @property
    def phase(self):
        """ The phase of a rotation gate. """
        return self.data

    def dagger(self):
        return type(self)(-self.phase)

    def grad(self, var):
        if len(self.dom) != 1:
            raise NotImplementedError
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        return scalar(np.pi * gradient) @ type(self)(self.phase + .5)


class Rx(Rotation):
    """ X rotations. """
    def __init__(self, phase):
        super().__init__(phase, name="Rx")

    @property
    def array(self):
        half_theta = self._pi * self.phase
        sin, cos = self._sin(half_theta), self._cos(half_theta)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]])


class Rz(Rotation):
    """ Z rotations. """
    def __init__(self, phase):
        super().__init__(phase, name="Rz")

    @property
    def array(self):
        half_theta = self._pi * self.phase
        return np.array(
            [[self._exp(-1j * half_theta), 0], [0, self._exp(1j * half_theta)]])


class Ry(Rotation):
    """ Y rotations. """
    def __init__(self, phase):
        super().__init__(phase, name="Ry")

    @property
    def array(self):
        half_theta = self._pi * self.phase
        sin, cos = self._sin(half_theta), self._cos(half_theta)
        return np.array([[cos, -1 * sin], [sin, cos]])


def _outer_prod_diag(*bitstring):
    return Bra(*bitstring) >> Ket(*bitstring)


class CU1(Rotation):
    """ Controlled Z rotations. """
    def __init__(self, phase):
        super().__init__(phase, name="CU1", n_qubits=2)

    @property
    def array(self):
        theta = 2 * self._pi * self.phase
        return np.array([1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, self._exp(1j * theta)])

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        _i_2_pi = 1j * 2 * self._pi
        return _outer_prod_diag(1, 1) @ scalar(_i_2_pi * gradient * self._exp(_i_2_pi * self.phase))


class CRz(Rotation):
    """ Controlled Z rotations. """
    def __init__(self, phase):
        super().__init__(phase, name="CRz", n_qubits=2)

    @property
    def array(self):
        half_theta = self._pi * self.phase
        return np.array([1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, self._exp(-1j * half_theta), 0,
                         0, 0, 0, self._exp(1j * half_theta)])

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient

        _i_half_pi = .5j * self._pi
        op1 = Z @ Z @ scalar(_i_half_pi * gradient)
        op2 = Id(qubit) @ Z @ scalar(-_i_half_pi * gradient)

        return self >> (op1 + op2)


class CRx(Rotation):
    """ Controlled Z rotations. """
    def __init__(self, phase):
        super().__init__(phase, name="CRx", n_qubits=2)

    @property
    def array(self):
        half_theta = self._pi * self.phase
        cos, sin = self._cos(half_theta), self._sin(half_theta)
        return np.array([1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, cos, -1j * sin,
                         0, 0, -1j * sin, cos])

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        
        _i_half_pi = .5j * self._pi
        op1 = Z @ X @ scalar(_i_half_pi * gradient)
        op2 = Id(qubit) @ X @ scalar(-_i_half_pi * gradient)

        return self >> (op1 + op2)


class Scalar(Parametrized):
    """ Scalar, i.e. quantum gate with empty domain and codomain. """
    def __init__(self, data, datatype=complex, name=None, is_mixed=False):
        self.drawing_name = format_number(data)
        name = "scalar" if name is None else name
        dom, cod = qubit ** 0, qubit ** 0
        _dagger = None if data.conjugate() == data else False
        super().__init__(
            name, dom, cod,
            datatype=datatype, is_mixed=is_mixed, data=data, _dagger=_dagger)

    def __repr__(self):
        return super().__repr__()[:-1] + (
            ', is_mixed=True)' if self.is_mixed else ')')

    @property
    def array(self):
        return [self.data]

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        return Scalar(self.array[0].diff(var))

    def dagger(self):
        return self if self._dagger is None\
            else Scalar(self.array[0].conjugate())


class MixedScalar(Scalar):
    """ Mixed scalar, i.e. where the Born rule has already been applied. """
    def __init__(self, data):
        super().__init__(data, is_mixed=True)


class Sqrt(Scalar):
    """ Square root. """
    def __init__(self, data):
        super().__init__(data, name="sqrt")
        self.drawing_name = "sqrt({})".format(format_number(data))

    @property
    def array(self):
        return [self.data ** .5]


def _shift_x(n, k):
    m = np.eye(n)
    m[k:k+2, k:k+2] = np.array([[0, 1], [1, 0]])
    return m


SWAP = Swap(qubit, qubit)
CX = QuantumGate('CX', 2, [1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 0, 1,
                           0, 0, 1, 0], _dagger=None)
CZ = QuantumGate('CZ', 2, [1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, -1], _dagger=None)
H = QuantumGate('H', 1, 1 / np.sqrt(2) * np.array([1, 1, 1, -1]), _dagger=None)
S = QuantumGate('S', 1, [1, 0, 0, 1j])
T = QuantumGate('T', 1, [1, 0, 0, np.exp(1j * np.pi / 4)])
X = QuantumGate('X', 1, [0, 1, 1, 0], _dagger=None)
Y = QuantumGate('Y', 1, [0, -1j, 1j, 0])
Z = QuantumGate('Z', 1, [1, 0, 0, -1], _dagger=None)

# CCX (Toffoli): |0><0| ⊗ I^{⊗2} + |1><1| ⊗ CX 
CCX = QuantumGate('CCX', 3, _shift_x(n=8, k=6))

# CSWAP (Fredkin): |0><0| ⊗ I^{⊗2} + |1><1| ⊗ SWAP
CSWAP = QuantumGate('CSWAP', 3, _shift_x(n=8, k=5))

GATES = [SWAP, CZ, CX, H, S, T, X, Y, Z, CCX, CSWAP]


def sqrt(expr):
    """ Returns a 0-qubit quantum gate that scales by a square root. """
    return Sqrt(expr)


def scalar(expr, is_mixed=False):
    """ Returns a 0-qubit quantum gate that scales by a complex number. """
    return Scalar(expr, is_mixed=is_mixed)
