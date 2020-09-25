import numpy as np

from discopy import rigid, Quiver
from discopy.rigid import Ob, Ty, Box, Diagram
from discopy.tensor import Dim, Tensor, TensorFunctor


class CQ(Ty):
    def __init__(self, classical=Dim(1), quantum=Dim(1)):
        self.classical, self.quantum = classical, quantum
        types = [Ob("C({})".format(dim)) for dim in classical]\
            + [Ob("Q({})".format(dim)) for dim in quantum]
        super().__init__(*types)

    def __repr__(self):
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


class CQMap(Box):
    def __init__(self, dom, cod, array):
        data = Tensor(dom.classical @ dom.quantum @ dom.quantum,
                      cod.classical @ cod.quantum @ cod.quantum, array)
        self.array = data.array
        super().__init__(array, dom, cod, data)

    def __eq__(self, other):
        return isinstance(other, CQMap)\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and self.data == other.data

    def __repr__(self):
        return "CQMap(dom={}, cod={}, array={})".format(
            self.dom, self.cod, self.array.flatten())

    @staticmethod
    def id(dom):
        data = Tensor.id(dom.classical @ dom.quantum @ dom.quantum)
        return CQMap(dom, dom, data.array)

    def then(self, other):
        data = self.data >> other.data
        return CQMap(self.dom, other.cod, data.array)

    def tensor(self, other):
        f = Box('f', Ty('c00', 'q00', 'q00'), Ty('c10', 'q10', 'q10'))
        g = Box('g', Ty('c01', 'q01', 'q01'), Ty('c11', 'q11', 'q11'))
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
            super().__init__(CQ(), CQ(), np.array(1))
        if len(dim) == 1:
            array = np.zeros(dim @ dim @ dim)
            for i in range(dim[0]):
                array[i, i, i] = 1
            return CQMap(Q(dim), C(dim), array)
        return CQMap.measure(dim[:1]) @ CQMap.measure(dim[1:])

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


class BitsAndQubits(Ty):
    def _upgrade(ty):
        return BitsAndQubits(ty.objects.count(Ob("bit")),
                             ty.objects.count(Ob("qubit")))

    def __init__(self, n_bits=0, n_qubits=0):
        self.n_bits, self.n_qubits = n_bits, n_qubits
        super().__init__(*(n_bits * [Ob("bit")] + n_qubits * [Ob("qubit")]))

    def __repr__(self):
        if not self.n_bits and not self.n_qubits:
            return "qubit ** 0"
        if not self.n_bits:
            return "qubit{}".format(
                " ** {}".format(self.n_qubits) if self.n_qubits > 1 else "")
        if not self.n_qubits:
            return "bit{}".format(
                " ** {}".format(self.n_bits) if self.n_bits > 1 else "")
        return "{} @ {}".format(
            BitsAndQubits(self.n_bits, 0), BitsAndQubits(0, self.n_qubits))

    def __str__(self):
        return repr(self)

    def tensor(self, other):
        if not isinstance(other, BitsAndQubits):
            return super().tensor(other)
        return BitsAndQubits(self.n_bits + other.n_bits,
                             self.n_qubits + other.n_qubits)


bit, qubit = BitsAndQubits(1, 0), BitsAndQubits(0, 1)


class CQCircuit(Diagram):
    @staticmethod
    def _upgrade(diagram):
        return CQCircuit(BitsAndQubits._upgrade(diagram.dom),
                         BitsAndQubits._upgrade(diagram.cod),
                         diagram.boxes, diagram.offsets, diagram.layers)

    def __init__(self, dom, cod, boxes, offsets, layers=None):
        super().__init__(dom, cod, boxes, offsets, layers)

    def __repr__(self):
        return super().__repr__().replace('Diagram', 'CQCircuit')

    @staticmethod
    def id(dom):
        return Id(dom)

    def eval(self):
        return EvalFunctor()(self)

    def pure_eval(self):
        return TensorFunctor({Ty('qubit'): 2}, Quiver(lambda g: g.array))(self)


class Id(rigid.Id, CQCircuit):
    def __init__(self, dom):
        rigid.Id.__init__(self, dom)
        CQCircuit.__init__(self, dom, dom, [], [])


class Discard(Box, CQCircuit):
    def __init__(self, dom=1):
        if isinstance(dom, int):
            dom = qubit ** dom
        Box.__init__(self, "Discard({})".format(dom), dom, qubit ** 0)
        CQCircuit.__init__(self, dom, qubit ** 0, [self], [0])


class Measure(Box, CQCircuit):
    def __init__(self, n_qubits=1):
        dom, cod = qubit ** n_qubits, bit ** n_qubits
        Box.__init__(self, "Measure({})".format(n_qubits), dom, cod)
        CQCircuit.__init__(self, dom, cod, [self], [0])


class Gate(Box, CQCircuit):
    def __init__(self, name, n_qubits, array=None):
        dom = qubit ** n_qubits
        if array is not None:
            self._array = np.array(array).reshape(2 * n_qubits * (2, ) or 1)
        Box.__init__(self, name, dom, dom)
        CQCircuit.__init__(self, dom, dom, [self], [0])

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "Gate({}, {}, {})".format(
            repr(self.name), len(self.dom), repr(self.array.flatten()))


class Ket(Box, CQCircuit):
    def __init__(self, *bitstring):
        self.bitstring = bitstring
        name = 'Ket({})'.format(', '.join(map(str, bitstring)))
        dom, cod = qubit ** 0, qubit ** len(bitstring)
        Box.__init__(self, name, dom, cod)
        CQCircuit.__init__(self, dom, cod, [self], [0])

    def tensor(self, other):
        if isinstance(other, Ket):
            return Ket(*(self.bitstring + other.bitstring))
        return super().tensor(other)

    def __repr__(self):
        return self.name

    @property
    def array(self):
        tensor = Tensor(Dim(1), Dim(1), [1])
        for bit in self.bitstring:
            tensor = tensor @ Tensor(Dim(2), Dim(1), [0, 1] if bit else [1, 0])
        return tensor.array


class Rx(Gate):
    def __init__(self, phase):
        self._phase = phase
        super().__init__('Rx', 1)

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


def sqrt(real):
    return Gate('sqrt({})'.format(real), 0, np.sqrt(real))


class EvalFunctor(rigid.Functor):
    def __init__(self):
        ob, ar = {Ty('bit'): C(Dim(2)), Ty('qubit'): Q(Dim(2))}, {}
        super().__init__(ob, ar, ob_factory=CQ, ar_factory=CQMap)

    def __call__(self, diagram):
        if isinstance(diagram, BitsAndQubits):
            return C(Dim(2) ** diagram.n_bits) @ Q(Dim(2) ** diagram.n_qubits)
        if isinstance(diagram, (Gate, Ket)):
            dom, cod = self(diagram.dom).quantum, self(diagram.cod).quantum
            return CQMap.pure(Tensor(dom, cod, diagram.array))
        if isinstance(diagram, Discard):
            return CQMap.discard(Q(self(diagram.dom).quantum))
        if isinstance(diagram, Measure):
            return CQMap.measure(self(diagram.dom).quantum)
        return super().__call__(diagram)


SWAP = Gate('SWAP', 2, [1, 0, 0, 0,
                        0, 0, 1, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 1])
CX = Gate('CX', 2, [1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 0])
CZ = Gate('CZ', 2, [1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, -1])
H = Gate('H', 1, 1 / np.sqrt(2) * np.array([1, 1, 1, -1]))
S = Gate('S', 1, [1, 0, 0, 1j])
T = Gate('T', 1, [1, 0, 0, np.exp(1j * np.pi / 4)])
X = Gate('X', 1, [0, 1, 1, 0])
Y = Gate('Y', 1, [0, -1j, 1j, 0])
Z = Gate('Z', 1, [1, 0, 0, -1])


def pure(circuit):
    return CQMap.pure(circuit.pure_eval())
