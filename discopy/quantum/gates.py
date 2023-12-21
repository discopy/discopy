# -*- coding: utf-8 -*-

"""
Gates in a :class:`discopy.quantum.circuit.Circuit`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    SelfConjugate
    AntiConjugate
    Discard
    MixedState
    Measure
    Encode
    QuantumGate
    ClassicalGate
    Copy
    Match
    Digits
    Bits
    Ket
    Bra
    Controlled
    Parametrized
    Rotation
    Rx
    Ry
    Rz
    CU1
    CRz
    CRx
    Scalar
    MixedScalar
    Sqrt

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        sqrt
        scalar
"""
import copy
from math import e, pi

from discopy import messages
from discopy.cat import rsubs
from discopy.matrix import get_backend
from discopy.quantum.circuit import (
    Circuit, Digit, Ty, bit, qubit, Box, Swap, Sum, Id)
from discopy.tensor import backend
from discopy.utils import factory_name, assert_isinstance


def format_number(data):
    """ Tries to format a number. """
    try:
        return f'{data:.3g}'
    except TypeError:
        return data


class SelfConjugate(Box):
    """ A self-conjugate box, i.e. where the transpose is the dagger. """
    def conjugate(self):
        return self

    def rotate(self, left=False):
        del left
        return self.dagger()


class AntiConjugate(Box):
    """ An anti-conjugate box, i.e. where the conjugate is the dagger. """
    def conjugate(self):
        return self.dagger()

    def rotate(self, left=False):
        del left
        return self


class Discard(SelfConjugate):
    """
    Discard n qubits. If :code:`dom == bit` then marginal distribution.
    """
    draw_as_discards = True

    def __init__(self, dom=1):
        if isinstance(dom, int):
            dom = qubit ** dom
        super().__init__(
            f"Discard({dom})", dom, qubit ** 0, is_mixed=True)
        self.n_qubits = len(dom)

    def dagger(self):
        return MixedState(self.dom)

    def _decompose(self):
        return Id().tensor(*[Discard()] * self.n_qubits)


class MixedState(SelfConjugate):
    """
    Maximally-mixed state on n qubits.
    If :code:`cod == bit` then uniform distribution.
    """
    draw_as_discards = True

    def __init__(self, cod=1):
        if isinstance(cod, int):
            cod = qubit ** cod
        super().__init__(
            f"MixedState({cod})", qubit ** 0, cod, is_mixed=True)
        self.drawing_name = "MixedState"
        if cod == bit:
            self.drawing_name = ""
            self.draw_as_spider, self.color = True, "black"

    def dagger(self):
        return Discard(self.cod)

    def _decompose(self):
        return Id().tensor(*[MixedState()] * len(self.cod))


class Measure(SelfConjugate):
    """
    Measure n qubits into n bits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits to measure.
    destructive : bool, optional
        Whether to do a non-destructive measurement instead.
    override_bits : bool, optional
        Whether to override input bits, this is the standard behaviour of tket.
    """
    draw_as_measures = True

    def __init__(self, n_qubits=1, destructive=True, override_bits=False):
        dom, cod = qubit ** n_qubits, bit ** n_qubits
        name = f"Measure({'' if n_qubits == 1 else n_qubits})"
        if not destructive:
            cod = qubit ** n_qubits @ cod
            name = name\
                .replace("()", "(1)").replace(')', ", destructive=False)")
        if override_bits:
            dom = dom @ bit ** n_qubits
            name = name\
                .replace("()", "(1)").replace(')', ", override_bits=True)")
        super().__init__(name, dom, cod, is_mixed=True)
        self.destructive, self.override_bits = destructive, override_bits
        self.n_qubits = n_qubits
        self.draw_as_measures = True

    def dagger(self):
        return Encode(self.n_qubits,
                      constructive=self.destructive,
                      reset_bits=self.override_bits)

    def _decompose(self):
        return Id().tensor(*[
            Measure(destructive=self.destructive,
                    override_bits=self.override_bits)] * self.n_qubits)


class Encode(SelfConjugate):
    """
    Controlled preparation, i.e. encode n bits into n qubits.

    Parameters
    ----------
    n_bits : int
        Number of bits to encode.
    constructive : bool, optional
        Whether to do a classically-controlled correction instead.
    reset_bits : bool, optional
        Whether to reset the bits to the uniform distribution.
    """
    draw_as_measures = True

    def __init__(self, n_bits=1, constructive=True, reset_bits=False):
        dom, cod = bit ** n_bits, qubit ** n_bits
        name = Measure(n_bits, constructive, reset_bits).name\
            .replace("Measure", "Encode")\
            .replace("destructive", "constructive")\
            .replace("override_bits", "reset_bits")
        super().__init__(name, dom, cod, is_mixed=True)
        self.constructive, self.reset_bits = constructive, reset_bits
        self.n_bits = n_bits

    def dagger(self):
        return Measure(self.n_bits,
                       destructive=self.constructive,
                       override_bits=self.reset_bits)

    def _decompose(self):
        return Id().tensor(*[
            Encode(constructive=self.constructive,
                   reset_bits=self.reset_bits)] * self.n_bits)


class QuantumGate(Box):
    """ Quantum gates, i.e. unitaries on n qubits. """
    is_mixed = False
    is_classical = False

    def __init__(self, name: str, dom: Ty, cod: Ty, data=None, **params):
        if data is not None and hasattr(data, "__len__"):
            data = [complex(v) for v in data]
        super().__init__(name, dom, cod, data, **params)

    def __setstate__(self, state):
        if "_array" in state and not state["_array"] is None:
            state["data"] = state['_array'].flatten().tolist()
        if "_name" in state:
            if state["_name"] in GATES and hasattr(
                    GATES[state["_name"]], "data"):
                state["data"] = copy.deepcopy(GATES[state["_name"]].data)
                state["_z"] = GATES[state["_name"]].z
        super().__setstate__(state)


class ClassicalGate(SelfConjugate):
    """
    Classical gates, i.e. from digits to digits.

    >>> from sympy import symbols
    >>> array = symbols("a b c d")
    >>> f = ClassicalGate('f', bit, bit, array)
    >>> f.data
    (a, b, c, d)
    >>> f.lambdify(*array)(1, 2, 3, 4).data
    (1, 2, 3, 4)
    """
    is_mixed = False
    is_classical = True


class Copy(ClassicalGate):
    """ Takes a bit, returns two copies of it. """
    def __init__(self):
        super().__init__("Copy", bit, bit ** 2, [1, 0, 0, 0, 0, 0, 0, 1])
        self.draw_as_spider, self.color = True, "black"
        self.drawing_name = ""

    def dagger(self):
        return Match()


class Match(ClassicalGate):
    """ Takes two bits in, returns them if they are equal. """
    def __init__(self):
        super().__init__("Match", bit ** 2, bit, [1, 0, 0, 0, 0, 0, 0, 1])
        self.draw_as_spider, self.color = True, "black"
        self.drawing_name = ""

    def dagger(self):
        return Copy()


class Digits(ClassicalGate):
    """
    Classical state for a given string of digits of a given dimension.

    Examples
    --------
    >>> from discopy.tensor import Dim, Tensor
    >>> assert Digits(2, dim=4).eval()\\
    ...     == Tensor[complex](dom=Dim(1), cod=Dim(4), array=[0, 0, 1, 0])
    """
    draw_as_brakets = True

    def __init__(self, *digits, dim=None, is_dagger=False):
        if not isinstance(dim, int):
            raise TypeError(int, dim)
        self._digits, self._dim = digits, dim
        str_digits = ', '.join(map(str, digits))
        name = f"Digits({str_digits}, dim={dim})" if dim != 2 \
            else f"Bits({str_digits})"
        dom, cod = Ty(), Ty(Digit(dim)) ** len(digits)
        dom, cod = (cod, dom) if is_dagger else (dom, cod)
        super().__init__(name, dom, cod, is_dagger=is_dagger)

    def __repr__(self):
        return self.name + (".dagger()" if self.is_dagger else "")

    @property
    def dim(self):
        """
        The dimension of the information units.

        >>> assert Bits(1, 0).dim == 2
        """
        return self._dim

    @property
    def digits(self):
        """ The digits of a classical state. """
        return list(self._digits)

    bitstring = digits

    @property
    def array(self):
        with backend('numpy') as np:
            array = np.zeros(len(self._digits) * (self._dim, ))
            array[self._digits] = 1
            return array

    def dagger(self):
        return Digits(*self.digits, dim=self.dim, is_dagger=not self.is_dagger)

    def to_drawing(self):
        result = QuantumGate.to_drawing(self)
        result.draw_as_brakets, result._digits = True, self._digits
        return result


def Bits(*bitstring, is_dagger=False):
    return Digits(*bitstring, dim=2, is_dagger=is_dagger)


class Ket(SelfConjugate, QuantumGate):
    """
    Implements qubit preparation for a given bitstring.

    >>> from discopy.tensor import Dim, Tensor
    >>> assert Ket(1, 0).cod == qubit ** 2
    >>> assert Ket(1, 0).eval()\\
    ...     == Tensor[complex](dom=Dim(1), cod=Dim(2, 2), array=[0, 0, 1, 0])
    """
    to_drawing = Digits.to_drawing
    array = Digits.array

    def __init__(self, *bitstring):
        if not all([bit in [0, 1] for bit in bitstring]):
            raise Exception('Bitstring can only contain integers 0 or 1.')

        dom, cod = qubit ** 0, qubit ** len(bitstring)
        name = f"Ket({', '.join(map(str, bitstring))})"
        super().__init__(name, dom, cod)
        self._digits, self._dim, self.draw_as_brakets = bitstring, 2, True

    @property
    def bitstring(self):
        """ The bitstring of a Ket. """
        return list(self._digits)

    def dagger(self):
        return Bra(*self.bitstring)

    def _decompose(self):
        return Id().tensor(*[Ket(b) for b in self.bitstring])


class Bra(SelfConjugate, QuantumGate):
    """
    Implements qubit post-selection for a given bitstring.

    >>> from discopy.tensor import Dim, Tensor
    >>> assert Bra(1, 0).dom == qubit ** 2
    >>> assert Bra(1, 0).eval()\\
    ...     == Tensor[complex](dom=Dim(2, 2), cod=Dim(1), array=[0, 0, 1, 0])
    """
    to_drawing = Digits.to_drawing
    array = Digits.array

    def __init__(self, *bitstring):
        if not all([bit in [0, 1] for bit in bitstring]):
            raise Exception('Bitstring can only contain integers 0 or 1.')

        name = f"Bra({', '.join(map(str, bitstring))})"
        dom, cod = qubit ** len(bitstring), qubit ** 0
        super().__init__(name, dom, cod)
        self._digits, self._dim, self.draw_as_brakets = bitstring, 2, True

    @property
    def bitstring(self):
        """ The bitstring of a Bra. """
        return list(self._digits)

    def dagger(self):
        return Ket(*self.bitstring)

    def _decompose(self):
        return Id().tensor(*[Bra(b) for b in self.bitstring])


class Controlled(QuantumGate):
    """
    Abstract class for controled quantum gates.

    Parameters
    ----------
    controlled : QuantumGate
        Gate to control, e.g. :code:`CX = Controlled(X)`.
    distance : int, optional
        Number of qubits from the control to the target, default is :code:`0`.
        If negative, the control is on the right of the target.
    """
    draw_as_controlled = True

    def __init__(self, controlled, distance=1):
        assert_isinstance(controlled, QuantumGate)
        if not distance:
            raise ValueError(messages.ZERO_DISTANCE_CONTROLLED)
        self.controlled, self.distance = controlled, distance
        n_qubits = len(controlled.dom) + abs(distance)
        name = f'C{controlled}'
        dom = cod = qubit ** n_qubits
        QuantumGate.__init__(self, name, dom, cod, data=controlled.data)

    def dagger(self):
        return Controlled(self.controlled.dagger(), distance=self.distance)

    def conjugate(self):
        controlled_conj = self.controlled.conjugate()
        return Controlled(controlled_conj, distance=-self.distance)

    def lambdify(self, *symbols, **kwargs):
        c_fn = self.controlled.lambdify(*symbols)
        return lambda *xs: type(self)(c_fn(*xs), distance=self.distance)

    def subs(self, *args):
        controlled = self.controlled.subs(*args)
        return type(self)(controlled, distance=self.distance)

    def __repr__(self):
        return f'Controlled({self.controlled!r}, distance={self.distance!r})'

    def __str__(self):
        return self.name if self.distance == 1\
            else f'Controlled({self.controlled}, distance={self.distance})'

    def __eq__(self, other):
        return not isinstance(other, Box) and super().__eq__(other)\
            or isinstance(other, Controlled)\
            and self.distance == other.distance\
            and self.controlled == other.controlled

    @property
    def phase(self):
        return self.controlled.phase

    __hash__ = QuantumGate.__hash__
    l = r = property(conjugate)

    def _decompose_grad(self):
        controlled, distance = self.controlled, self.distance
        if isinstance(controlled, (Rx, Rz)):
            phase = self.phase
            decomp = Controlled(X, distance=distance)\
                >> qubit ** distance @ Rz(-phase / 2) @ qubit ** -distance\
                >> Controlled(X, distance=distance)\
                >> qubit ** distance @ Rz(phase / 2) @ qubit ** -distance
            if isinstance(controlled, Rx):
                decomp <<= qubit ** distance @ H @ qubit ** -distance
                decomp >>= qubit ** distance @ H @ qubit ** -distance
            return decomp
        return self

    def _decompose(self):
        controlled, distance = self.controlled, self.distance
        n_qubits = len(self.dom)

        if distance == 1:
            return self

        skipped_qbs = n_qubits - (1 + len(controlled.dom))

        if distance > 0:
            pattern = [0,
                       *range(skipped_qbs + 1, n_qubits),
                       *range(1, skipped_qbs + 1)]
        else:
            pattern = [n_qubits - 1, *range(n_qubits - 1)]

        perm = Circuit.permutation(pattern)
        diagram = (perm
                   >> type(self)(controlled) @ Id(skipped_qbs)
                   >> perm[::-1])

        return diagram

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        decomp = self._decompose_grad()
        if decomp == self:
            raise NotImplementedError()
        return decomp.grad(var, **params)

    @property
    def array(self):
        controlled, distance = self.controlled, self.distance
        n_qubits = len(self.dom)
        with backend() as np:
            if distance == 1:
                d = 1 << n_qubits - 1
                part1 = np.array([[1, 0], [0, 0]])
                part2 = np.array([[0, 0], [0, 1]])
                array = np.kron(part1, np.eye(d))\
                    + np.kron(part2, np.array(controlled.array.reshape(d, d)))
            else:
                array = self._decompose().eval().array
        return array.reshape(*[2] * 2 * n_qubits)

    def to_drawing(self):
        result = super().to_drawing()
        result.distance, result.controlled = self.distance, self.controlled
        return result


class Parametrized(Box):
    """
    Abstract class for parametrized boxes in a quantum circuit.

    Parameters
    ----------
    name : str
        Name of the parametrized class, e.g. :code:`"CRz"`.
    dom, cod : discopy.quantum.circuit.Ty
        Domain and codomain.
    data : any
        Data of the box, potentially with free symbols.

    Example
    -------
    >>> from sympy.abc import phi
    >>> from sympy import pi, exp, I
    >>> assert Rz(phi).array[0,0] == exp(-1.0 * I * pi * phi)
    >>> c = Rz(phi) >> Rz(-phi)
    >>> assert c.lambdify(phi)(.25) == Rz(.25) >> Rz(-.25)
    """
    def __init__(self, name, dom, cod, data=None, **params):
        self.drawing_name = f'{name}({data})'
        Box.__init__(self, name, dom, cod, data=data, **params)

    @property
    def modules(self):
        if self.free_symbols:
            import sympy
            return sympy
        else:
            return get_backend()

    def subs(self, *args):
        data = rsubs(self.data, *args)
        return type(self)(data)

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify
        with backend() as np:
            data = lambdify(symbols, self.data, dict(kwargs, modules=np))
        return lambda *xs: type(self)(data(*xs))

    def __str__(self):
        if isinstance(self, Controlled):
            # Ensure `Controlled(Rx(0.5))` and `CRx(0.5)` are printed the same.
            return Controlled.__str__(self)
        return f'{self.name}({format_number(self.data)})'

    def __repr__(self):
        return factory_name(type(self)) + f"({format_number(self.data)})"


class Rotation(Parametrized, QuantumGate):
    """ Abstract class for rotation gates. """
    n_qubits = 1

    def __init__(self, phase, z=0):
        name, n_qubits = type(self).__name__, type(self).n_qubits
        dom = cod = qubit ** n_qubits
        QuantumGate.__init__(self, name, dom, cod, z=z)
        Parametrized.__init__(self, name, dom, cod, is_mixed=False, data=phase)

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(tree['data'], tree.get('z', 0))

    @property
    def phase(self):
        """ The phase of a rotation gate. """
        return self.data

    def dagger(self):
        return type(self)(-self.phase)

    def rotate(self, left=False):
        del left
        return type(self)(self.phase, z=int(not self.z))

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient

        with backend() as np:
            if params.get('mixed', True):
                if len(self.dom) != 1:
                    raise NotImplementedError
                s = scalar(np.pi * gradient, is_mixed=True)
                t1 = type(self)(self.phase + .25)
                t2 = type(self)(self.phase - .25)
                return s @ (t1 + scalar(-1, is_mixed=True) @ t2)

            return scalar(np.pi * gradient) @ type(self)(self.phase + .5)


class Rx(AntiConjugate, Rotation):
    """ X rotations. """
    @property
    def array(self):
        with backend() as np:
            half_theta = np.array(self.modules.pi * self.phase, dtype=complex)
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)
            return np.stack((cos, -1j * sin, -1j * sin, cos)).reshape(2, 2)


class Ry(SelfConjugate, Rotation):
    """ Y rotations. """
    @property
    def array(self):
        with backend() as np:
            half_theta = np.array(self.modules.pi * self.phase)
            sin = self.modules.sin(half_theta)
            cos = self.modules.cos(half_theta)
            return np.stack((cos, sin, -sin, cos)).reshape(2, 2)


class Rz(AntiConjugate, Rotation):
    """ Z rotations. """
    @property
    def array(self):
        with backend() as np:
            half_theta = np.array(self.modules.pi * self.phase)
            e1 = self.modules.exp(-1j * half_theta)
            e2 = self.modules.exp(1j * half_theta)
            z = np.array(0)
            return np.stack((e1, z, z, e2)).reshape(2, 2)


class U1(AntiConjugate, Rotation):
    """ Z rotation, differ from :class:`Rz` by a global phase. """
    @property
    def array(self):
        with backend() as np:
            theta = np.array(2 * self.modules.pi * self.phase)
            return np.stack(
                (1, 0, 0, self.modules.exp(1j * theta))).reshape(2, 2)


class ControlledRotation(Controlled, Rotation):
    """ Controlled rotation gate. """
    def __init__(self, phase, distance=1):
        Controlled.__init__(self, self.controlled(phase), distance)

    lambdify = Rotation.lambdify
    subs = Rotation.subs


class CU1(ControlledRotation):
    """ Controlled U1 rotations. """
    controlled = U1


class CRz(ControlledRotation):
    """ Controlled Z rotations. """
    controlled = Rz


class CRx(ControlledRotation):
    """ Controlled X rotations. """
    controlled = Rx


class Scalar(Parametrized):
    """ Scalar, i.e. quantum gate with empty domain and codomain. """
    def __init__(self, data, name=None, is_mixed=False):
        self.drawing_name = format_number(data)
        name = "scalar" if name is None else name
        dom, cod = qubit ** 0, qubit ** 0
        super().__init__(name, dom, cod, is_mixed=is_mixed, data=data, z=None)

    def __setstate__(self, state):
        state["_z"] = None
        super().__setstate__(state)

    def __repr__(self):
        return super().__repr__()[:-1] + (
            ', is_mixed=True)' if self.is_mixed else ')')

    @property
    def array(self):
        with backend() as np:
            return np.array(self.data)

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        return Scalar(self.data.diff(var))

    def dagger(self):
        return Scalar(self.data.conjugate(), self.name, self.is_mixed)


class MixedScalar(Scalar):
    """ Mixed scalar, i.e. where the Born rule has already been applied. """
    def __init__(self, data):
        super().__init__(data, is_mixed=True)


class Sqrt(Scalar):
    """ Square root. """
    def __init__(self, data):
        super().__init__(data, name="sqrt")
        self.drawing_name = f"sqrt({format_number(data)})"

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.is_dagger is None:
            self.is_dagger = False

    @property
    def array(self):
        with backend() as np:
            return np.array(self.data ** .5)

    def dagger(self):
        return self


def sqrt(expr):
    """ Returns a 0-qubit quantum gate that scales by a square root. """
    return Sqrt(expr)


def scalar(expr, is_mixed=False):
    """ Returns a 0-qubit quantum gate that scales by a complex number. """
    return Scalar(expr, is_mixed=is_mixed)


SWAP = Swap(qubit, qubit)
H = QuantumGate(
    'H', qubit, qubit,
    data=[2 ** -0.5 * x for x in [1, 1, 1, -1]], is_dagger=None, z=None)
S = QuantumGate('S', qubit, qubit, [1, 0, 0, 1j])
T = QuantumGate('T', qubit, qubit, [1, 0, 0, e ** (1j * pi / 4)])
X = QuantumGate('X', qubit, qubit, [0, 1, 1, 0], is_dagger=None, z=None)
Y = QuantumGate('Y', qubit, qubit, [0, 1j, -1j, 0], is_dagger=None)
Z = QuantumGate('Z', qubit, qubit, [1, 0, 0, -1], is_dagger=None, z=None)
CX = Controlled(X)
CY = Controlled(Y)
CZ = Controlled(Z)
CCX = Controlled(CX)
CCZ = Controlled(CZ)

GATES = {
    'SWAP': SWAP,
    'H': H,
    'S': S,
    'T': T,
    'X': X,
    'Y': Y,
    'Z': Z,
    'CZ': CZ,
    'CY': CY,
    'CX': CX,
    'CCX': CCX,
    'CCZ': CCZ,
    'Rx': Rx,
    'Ry': Ry,
    'Rz': Rz,
    'U1': U1,
    'CRx': CRx,
    'CRz': CRz,
    'CU1': CU1,
}


for attr, gate in GATES.items():
    def closure(attr=attr, gate=gate):
        """ Easiest way around the Python late binding gotcha. """
        if isinstance(gate, Controlled)\
                and isinstance(gate.controlled, Controlled):
            def method(self, i: int, j: int, k: int) -> Circuit:
                """
                Apply {} gate to a circuit given qubit indices.

                Parameters:
                    i : First control index.
                    j : Second control index.
                    k : Target index.
                """
                return self.apply_controlled(
                    gate.controlled.controlled, i, j, k)
        elif isinstance(gate, Controlled):
            def method(self, i: int, j: int) -> Circuit:
                """
                Apply {} gate to a circuit given qubit indices.

                Parameters:
                    i : Control index.
                    j : Target index.
                """
                return self.apply_controlled(gate.controlled, i, j)
        elif isinstance(gate, Box):
            def method(self, i: int) -> Circuit:
                """
                Apply {} gate to a circuit given qubit index.

                Parameters:
                    i : Target index.
                """
                return self.apply_controlled(gate, i)
        elif issubclass(gate, Rotation) and issubclass(gate, Controlled):
            def method(self, phi: float, i: int, j: int) -> Circuit:
                """
                Apply :class:`{}` to a circuit given phase and indices.

                Parameters:
                    phi : Phase.
                    i : Control index.
                    j : Target index.
                """
                return self.apply_controlled(gate.controlled(phi), i, j)
        elif issubclass(gate, Rotation):
            def method(self, phi: float, i: int) -> Circuit:
                """
                Apply :class:`{}` to a circuit given phase and target index.

                Parameters:
                    phi : Phase.
                    i : Target index.
                """
                return self.apply_controlled(gate(phi), i)
        method.__doc__ = method.__doc__.format(attr)
        return method
    setattr(Circuit, attr, closure())
