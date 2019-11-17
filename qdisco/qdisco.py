import numpy as np
from discopy.cat import Quiver
from discopy.moncat import Ob, Ty, Diagram, Box, MonoidalFunctor
from discopy.matrix import MatrixFunctor, Dim
from discopy.disco import Adjoint, Pregroup, Grammar, Wire, Word, Model
from circuit import CircuitFunctor, Circuit, Gate, PRO, Id, GCX, Permutation, H

Dummy = lambda x: Pregroup('dummy{}'.format(x))

class QCup(Box):
    def __init__(self, x, dagger=False):
        if isinstance(x, Pregroup):
            assert len(x) == 1
            x = x[0]
        elif not isinstance(x, Adjoint):
            x = Adjoint(x, 0)
        dom, cod = Pregroup(x, x.l) if dagger else Pregroup(x, x.r), Dummy(x._basic)
        Box.__init__(self, 'qcup_{}'.format(x), dom, cod, dagger)

    def dagger(self):
        return QCap(self.dom[0], not self._dagger)

    def __repr__(self):
        return "QCup({}{})".format(repr(
            self.dom[0] if self.dom[0]._z else self.dom[0]._basic),
            ", dagger=True" if self._dagger else "")

    def __str__(self):
        return "QCup({}{})".format(str(self.dom[0]),
                                ", dagger=True" if self._dagger else "")

class QCap(Box):
    def __init__(self, x, dagger=False):
        if isinstance(x, Pregroup):
            assert len(x) == 1
            x = x[0]
        elif not isinstance(x, Adjoint):
            x = Adjoint(x, 0)
        dom, cod = Dummy(x._basic), Pregroup(x, x.r) if dagger else Pregroup(x, x.l)
        Box.__init__(self, 'cap_{}'.format(x), dom, cod, dagger)

    def dagger(self):
        return QCup(self.cod[0], not self._dagger)

    def __repr__(self):
        return "QCap({}{})".format(repr(
            self.cod[0] if self.cod[0]._z else self.cod[0]._basic),
            ", dagger=True" if self._dagger else "")

    def __str__(self):
        return "QCap({}{})".format(str(self.cod[0]),
                                ", dagger=True" if self._dagger else "")

class QParse(Diagram):
    def __init__(self, words, cups):
        self._words, self._cups = words, cups
        self._type = sum((w.type for w in words), Pregroup())
        dom = sum((w.dom for w in words), Pregroup())
        boxes = words[::-1]  # words are backwards to make offsets easier
        offsets = [len(words) - i - 1 for i in range(len(words))]
        cod = self._type
        for x in range(len(cups)):
            i = cups[x]
            assert cod[i].r == cod[i + 1]
            boxes.append(QCup(cod[i]))
            offsets.append(i)
            cod = cod[:i] + Dummy(cod[i]._basic) + cod[i+2:]
            if x < len(cups) -1 and cups[x + 1]  == cups[x] - 1:
                boxes.append(GSWAP(Pregroup(cod[i]), Pregroup(cod[i+1])))
                offsets.append(i)
                cod = cod[:i] + [cod[i+1], cod[i]] + cod[i+2:]
        super().__init__(dom, cod, boxes, offsets)

    def __str__(self):
        return "{} >> {}".format(" @ ".join(self._words), str(Diagram(self._type,
            self.cod, self.boxes[len(self._words) + 1:], self._cups)))

class GSWAP(Box):
    def __init__(self, dom0, dom1):
        assert isinstance(dom0, Ty)
        assert isinstance(dom1, Ty)
        self.dom0, self.dom1 = dom0, dom1
        dom = dom0 + dom1
        cod = dom1 + dom0
        Box.__init__(self, 'GSWAP', dom, cod)

    def __repr__(self):
        return "GSWAP({}, {})".format(repr(self.dom0), repr(self.dom1))

    def __str__(self):
        return "GSWAP({}, {})".format(str(self.dom0), str(self.dom1))

class CircuitModel(CircuitFunctor):
    def __init__(self, ob, ar):
        self._ob, self._ar = ob, ar

    def __repr__(self):
        return "CircuitModel(ob={}, ar={})".format(self._types, self._vocab)

    def __call__(self, d):
        if isinstance(d, Adjoint):
            return self.ob[Pregroup(d._basic)]
        if isinstance(d, Pregroup):
            n_qubits = sum([self.__call__(b) for b in d], Ty())
            return n_qubits
        return super().__call__(d)

# We can encode the language of a pregroup grammar into QParses

s, n = Pregroup('s'), Pregroup('n')
B = [s, n]

Alice, Bob, jokes = Word('Alice', n), Word('Bob', n), Word('jokes', n)
loves, tells = Word('loves', n.r + s + n.l), Word('tells', n.r + s + n.l)
who = Word('who', n.r + n.l.r + s.l + n)
vocab = [Alice, Bob, jokes, loves, tells, jokes, who]

parse0 = QParse([Alice, loves, Bob], [0, 2])
parse1 = QParse([Alice, loves, Bob, who, tells, jokes],
                [0,  3, 2, 5, 4, 6])
parse2 = QParse([Bob, tells, jokes], [0, 2])

# We can translate parses from disco.py to qparses

def parse_to_qparse(cups):
    dummy_count = 0
    qcups = [cups[0]]
    for i in range(1, len(cups)):
        dummy_count = i
        if cups[i - 1] == cups[i] + 1:
            qcups += [cups[i] + dummy_count - 1]
        else:
            qcups += [cups[i] + dummy_count]
    return qcups

assert parse_to_qparse([0,1]) == [0,2] == parse0._cups
assert parse_to_qparse([0, 2, 1, 2, 1, 1]) == parse1._cups

# We can map QParses to Circuits

ob = {s: PRO(0), n: PRO(1)}
ob.update({Dummy(x): ob[x] + ob[x] for x in B})
Type_to_Pro = CircuitModel(ob, {})
ob.update({Pregroup(w.word): Type_to_Pro(w.type) for w in vocab})
Type_to_Pro = CircuitModel(ob, {})

def qparse_to_circuit(d):
    if isinstance(d, QCup):
        n = int(len(Type_to_Pro(d.dom)) / 2)
        HAD = Circuit(0, [], [])
        for i in range(n):
            HAD = HAD @ H
        HAD = HAD @ Circuit.id(n)
        return GCX(n) >> HAD
    if isinstance(d, QCap):
        n = int(len(Type_to_Pro(d.cod)) / 2)
        HAD = Circuit(0, [], [])
        for i in range(n):
            HAD = HAD @ H
        HAD = HAD @ Circuit.id(n)
        return GCX(n) << HAD
    if isinstance(d, GSWAP):
        n_qubits0 = len(Type_to_Pro(d.dom0))
        n_qubits1 = len(Type_to_Pro(d.dom1))
        n_qubits = n_qubits0 + n_qubits1
        gates = [Gate('SWAP', 2) for i in range(n_qubits0 * n_qubits1)]
        offsets = []
        for i in range(n_qubits1):
            offsets += range(i, n_qubits0 + i)[::-1]
        return Circuit(n_qubits, gates, offsets)
    elif isinstance(d, Word):
        n_qubits = len(Type_to_Pro(d.type))
        return Gate(d.word, n_qubits)

ar = Quiver(qparse_to_circuit)
F = CircuitModel(ob, ar)
circuit0 = F(parse0)
circuit1 = F(parse1)
circuit2 = F(parse2)
