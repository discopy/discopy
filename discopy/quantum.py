from itertools import takewhile

from discopy import rigid, Quiver
from discopy.rigid import Ob, Ty, Diagram
from discopy.tensor import np, Dim, Tensor, TensorFunctor


class CQ(Ty):
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
        return "C({}) @ Q({})".format(repr(self.quantum), repr(self.classical))

    def __str__(self):
        return repr(self)

    def tensor(self, other):
        return CQ(
            self.classical @ other.classical, self.quantum @ other.quantum)


class C(CQ):
    def __init__(self, dim):
        super().__init__(dim, Dim(1))


class Q(CQ):
    def __init__(self, dim):
        super().__init__(Dim(1), dim)


class CQMap(rigid.Box):
    def __init__(self, dom, cod, array):
        data = Tensor(dom.classical @ dom.quantum @ dom.quantum,
                      cod.classical @ cod.quantum @ cod.quantum, array)
        self.array = data.array
        super().__init__("CQMap", dom, cod, data=data)

    def __eq__(self, other):
        return isinstance(other, CQMap)\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and self.data == other.data

    def __repr__(self):
        return "CQMap(dom={}, cod={}, array={})".format(
            self.dom, self.cod, np.array2string(self.array.flatten()))

    def __str__(self):
        return repr(self)

    @staticmethod
    def id(dom):
        data = Tensor.id(dom.classical @ dom.quantum @ dom.quantum)
        return CQMap(dom, dom, data.array)

    def then(self, other):
        data = self.data >> other.data
        return CQMap(self.dom, other.cod, data.array)

    def tensor(self, other):
        f = rigid.Box('f', Ty('c00', 'q00', 'q00'), Ty('c10', 'q10', 'q10'))
        g = rigid.Box('g', Ty('c01', 'q01', 'q01'), Ty('c11', 'q11', 'q11'))
        ob = {Ty("{}{}{}".format(a, b, c)):
              z.__getattribute__(y).__getattribute__(x)
              for a, x in zip(['c', 'q'], ['classical', 'quantum'])
              for b, y in zip([0, 1], ['dom', 'cod'])
              for c, z in zip([0, 1], [self, other])}
        ar = {f: self.array, g: other.array}
        permute_above = Diagram.id(f.dom[:1] @ g.dom[:1] @ f.dom[1:2])\
            @ Diagram.swap(g.dom[1:2], f.dom[2:]) @ Diagram.id(g.dom[2:])\
            >> Diagram.id(f.dom[:1]) @ Diagram.swap(g.dom[:1], f.dom[1:])\
            @ Diagram.id(g.dom[1:])
        permute_below =\
            Diagram.id(f.cod[:1]) @ Diagram.swap(f.cod[1:], g.cod[:1])\
            @ Diagram.id(g.cod[1:])\
            >> Diagram.id(f.cod[:1] @ g.cod[:1] @ f.cod[1:2])\
            @ Diagram.swap(f.cod[2:], g.cod[1:2]) @ Diagram.id(g.cod[2:])
        F = TensorFunctor(ob, ar)
        array = F(permute_above >> f @ g >> permute_below).array
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        return CQMap(dom, cod, array)

    @staticmethod
    def measure(dim):
        if not dim:
            return CQMap(CQ(), CQ(), np.array(1))
        if len(dim) == 1:
            array = np.array([
                i == j == k
                for i in range(dim[0])
                for j in range(dim[0])
                for k in range(dim[0])])
            return CQMap(Q(dim), C(dim), array)
        return CQMap.measure(dim[:1]) @ CQMap.measure(dim[1:])

    @staticmethod
    def encode(dim):
        return CQMap(C(dim), Q(dim), CQMap.measure(dim).array)

    @staticmethod
    def pure(process):
        return CQMap(Q(process.dom), Q(process.cod),
                     (process.conjugate() @ process).array)

    @staticmethod
    def discard(dom):
        array = np.tensordot(
            np.ones(dom.classical), Tensor.id(dom.quantum).array, 0)
        return CQMap(dom, CQ(), array)

    def is_close(self, other):
        return isinstance(other, CQMap)\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and np.allclose(self.array, other.array)

    @property
    def is_causal(self):
        return self.discard(self.dom).is_close(self >> self.discard(self.cod))

    def cups(left, right):
        assert not left.classical and not right.classical
        return pure(Tensor.cups(left.quantum, right.quantum))

    def caps(left, right):
        assert not left.classical and not right.classical
        return pure(Tensor.caps(left.quantum, right.quantum))


class BitsAndQubits(Ty):
    def __repr__(self):
        if not self:
            return "Ty()"
        n_bits = len(list(takewhile(lambda x: x.name == "bit", self)))
        n_qubits = len(list(takewhile(
            lambda x: x.name == "qubit", self[n_bits:])))
        remainder = self[n_bits + n_qubits:]
        left = "" if not n_bits else "bit{}".format(
            " ** {}".format(n_bits) if n_bits > 1 else "")
        middle = "" if not n_qubits else "qubit{}".format(
            " ** {}".format(n_qubits) if n_qubits > 1 else "")
        right = "" if not remainder else repr(BitsAndQubits(*remainder))
        return " @ ".join(s for s in [left, middle, right] if s)

    def __str__(self):
        return repr(self)


bit, qubit = BitsAndQubits("bit"), BitsAndQubits("qubit")


class CQCircuit(Diagram):
    @staticmethod
    def _upgrade(diagram):
        dom = BitsAndQubits(*diagram.dom.objects)
        cod = BitsAndQubits(*diagram.cod.objects)
        return CQCircuit(
            dom, cod, diagram.boxes, diagram.offsets, diagram.layers)

    def __repr__(self):
        return super().__repr__().replace('Diagram', 'CQCircuit')

    @staticmethod
    def id(dom):
        return Id(dom)

    def eval(self):
        ob, ar = {Ty('bit'): C(Dim(2)), Ty('qubit'): Q(Dim(2))}, {}
        return Functor(ob, ar)(self)

    def pure_eval(self):
        return TensorFunctor({Ty('qubit'): 2}, Quiver(lambda g: g.array))(self)


class Id(rigid.Id, CQCircuit):
    def __init__(self, dom):
        rigid.Id.__init__(self, dom)
        CQCircuit.__init__(self, dom, dom, [], [])


class Box(rigid.Box, CQCircuit):
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        rigid.Box.__init__(self, name, dom, cod, data=data, _dagger=_dagger)
        CQCircuit.__init__(self, dom, cod, [self], [0])

    def __repr__(self):
        return self.name


class Discard(Box):
    def __init__(self, dom=1):
        if isinstance(dom, int):
            dom = qubit ** dom
        super().__init__("Discard({})".format(dom), dom, qubit ** 0)

    def dagger(self):
        return MixedState(self.dom)


class MixedState(Box):
    def __init__(self, cod=1):
        if isinstance(cod, int):
            cod = qubit ** cod
        super().__init__("MixedState({})".format(cod), qubit ** 0, cod)

    def dagger(self):
        return Discard(self.cod)


class Measure(Box):
    def __init__(self, n_qubits=1):
        dom, cod = qubit ** n_qubits, bit ** n_qubits
        super().__init__("Measure({})".format(n_qubits), dom, cod)

    def dagger(self):
        return Encode(len(self.cod))


class Encode(Box):
    def __init__(self, n_bits=1):
        dom, cod = bit ** n_bits, qubit ** n_bits
        super().__init__("Encode({})".format(n_bits), dom, cod)

    def dagger(self):
        return Measure(len(self.dom))


class QGate(Box):
    def __init__(self, name, n_qubits, array=None, _dagger=False):
        dom = qubit ** n_qubits
        if array is not None:
            self._array = np.array(array).reshape(2 * n_qubits * (2, ) or 1)
        super().__init__(name, dom, dom, _dagger=_dagger)

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "QGate({}, {}, {})".format(
            repr(self.name), len(self.dom),
            np.array2string(self.array.flatten()))

    def dagger(self):
        return QGate(
            self.name, len(self.dom), self.array,
            _dagger=None if self._dagger is None else not self._dagger)


class CGate(Box):
    def __init__(self, name, n_bits_in, n_bits_out, array, _dagger=False):
        dom, cod = bit ** n_bits_in, bit ** n_bits_out
        if array is not None:
            self._array = np.array(array).reshape(
                (n_bits_in + n_bits_out) * (2, ) or 1)
        super().__init__(name, dom, cod, _dagger=_dagger)

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "CGate({}, {}, {}, {})".format(
            repr(self.name), len(self.dom), len(self.cod),
            np.array2string(self.array))

    def dagger(self):
        return CGate(
            self.name, len(self.dom), len(self.cod), self.array,
            _dagger=None if self._dagger is None else not self._dagger)

    @staticmethod
    def func(name, n_bits_in, n_bits_out, function):
        array = np.zeros((n_bits_in + n_bits_out) * [2])
        for i in range(2 ** n_bits_in):
            bitstring = tuple(map(int, format(i, '0{}b'.format(n_bits_in))))
            array[bitstring + tuple(function(*bitstring))] = 1
        return CGate(name, n_bits_in, n_bits_out, array)


class Ket(Box):
    def __init__(self, *bitstring):
        self.bitstring = bitstring
        name = 'Ket({})'.format(', '.join(map(str, bitstring)))
        dom, cod = qubit ** 0, qubit ** len(bitstring)
        super().__init__(name, dom, cod)

    @property
    def array(self):
        tensor = Tensor(Dim(1), Dim(1), [1])
        for bit in self.bitstring:
            tensor = tensor @ Tensor(Dim(2), Dim(1), [0, 1] if bit else [1, 0])
        return tensor.array

    def dagger(self):
        return Bra(*self.bitstring)


class Bra(Box):
    def __init__(self, *bitstring):
        self.bitstring = bitstring
        name = 'Bra({})'.format(', '.join(map(str, bitstring)))
        dom, cod = qubit ** len(bitstring), qubit ** 0
        super().__init__(name, dom, cod)

    @property
    def array(self):
        return Ket(*self.bitstring).array

    def dagger(self):
        return Ket(*self.bitstring)


class Rx(QGate):
    def __init__(self, phase):
        self._phase = phase
        super().__init__('Rx', 1, array=None)

    @property
    def phase(self):
        return self._phase

    @property
    def name(self):
        return 'Rx({})'.format(self.phase)

    def __repr__(self):
        return self.name

    @property
    def array(self):
        half_theta = np.pi * self.phase
        global_phase = np.exp(1j * half_theta)
        sin, cos = np.sin(half_theta), np.cos(half_theta)
        return global_phase * np.array([[cos, -1j * sin], [-1j * sin, cos]])

    def dagger(self):
        return Rx(-self.phase)


class Functor(rigid.Functor):
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=CQ, ar_factory=CQMap)

    def __call__(self, diagram):
        if isinstance(diagram, (QGate, Bra, Ket)):
            dom, cod = self(diagram.dom).quantum, self(diagram.cod).quantum
            return CQMap.pure(Tensor(dom, cod, diagram.array))
        if isinstance(diagram, CGate):
            dom, cod = self(diagram.dom), self(diagram.cod)
            return CQMap(dom, cod, diagram.array)
        if isinstance(diagram, Discard):
            return CQMap.discard(self(diagram.dom))
        if isinstance(diagram, Measure):
            return CQMap.measure(self(diagram.dom).quantum)
        if isinstance(diagram, Encode):
            return CQMap.encode(self(diagram.dom).classical)
        return super().__call__(diagram)


SWAP = QGate('SWAP', 2, [1, 0, 0, 0,
                         0, 0, 1, 0,
                         0, 1, 0, 0,
                         0, 0, 0, 1], _dagger=None)
CX = QGate('CX', 2, [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 1, 0], _dagger=None)
CZ = QGate('CZ', 2, [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, -1], _dagger=None)
H = QGate('H', 1, 1 / np.sqrt(2) * np.array([1, 1, 1, -1]), _dagger=None)
S = QGate('S', 1, [1, 0, 0, 1j])
T = QGate('T', 1, [1, 0, 0, np.exp(1j * np.pi / 4)])
X = QGate('X', 1, [0, 1, 1, 0], _dagger=None)
Y = QGate('Y', 1, [0, -1j, 1j, 0])
Z = QGate('Z', 1, [1, 0, 0, -1], _dagger=None)


def sqrt(real):
    return QGate('sqrt({})'.format(real), 0, np.sqrt(real), _dagger=None)


def scalar(complex):
    return Gate('scalar({:.3f})'.format(complex), 0, complex,
                _dagger=None if np.conjugate(complex) == complex else False)
