# -*- coding: utf-8 -*-

"""
Implements the translation between discopy and pytket.
"""

from warnings import warn

import pytket as tk
from pytket import *
from pytket.circuit import Bit, Qubit
from pytket.utils import probs_from_counts

from discopy import messages
from discopy.cat import Quiver
from discopy.tensor import np, Dim, Tensor
from discopy.quantum import (
    CircuitFunctor, Id, Bra, Ket, BitsAndQubits, Swap, Ob,
    bit, qubit, Discard, Measure, gates, SWAP, X, Rx, Rz, CRz, scalar)


class Circuit(tk.Circuit):
    """
    pytket.Circuit with post selection and scalars.
    """
    def __init__(self, n_qubits=0, n_bits=0, post_selection=None, scalar=None):
        self.post_selection = post_selection or {}
        self.scalar = scalar or 1
        super().__init__(n_qubits, n_bits)

    def __repr__(self):
        def repr_gate(gate):
            name, inputs = gate.op.type.name, gate.op.params + [
                x.index[0] for x in gate.qubits + gate.bits]
            return "{}({})".format(name, ", ".join(map(str, inputs)))
        init = ["tk.Circuit({}{})".format(
            self.n_qubits, ", {}".format(self.n_bits) if self.bits else "")]
        gates = list(map(repr_gate, list(self)))
        post_select = ["post_select({})".format(self.post_selection)]\
            if self.post_selection else []
        scalar = ["scale({})".format(x) for x in [self.scalar] if x != 1]
        return '.'.join(init + gates + post_select + scalar)

    @property
    def n_bits(self):
        return len(self.bits)

    def rename_units(self, renaming):
        bits_to_rename = [old for old in renaming.keys()
                          if isinstance(old, Bit)
                          and old.index[0] in self.post_selection]
        post_selection_renaming = {
            renaming[old].index[0]: self.post_selection[old.index[0]]
            for old in bits_to_rename}
        for old in bits_to_rename:
            del self.post_selection[old.index[0]]
        self.post_selection.update(post_selection_renaming)
        super().rename_units(renaming)

    def scale(self, number):
        self.scalar *= number
        return self

    def post_select(self, post_selection):
        self.post_selection.update(post_selection)
        return self


def to_tk(circuit):
    if circuit.dom != qubit ** len(circuit.dom):
        raise ValueError("Circuit should have qubits as domains.")
    if circuit.dom:
        circuit = Ket(*(len(circuit.dom) * [0])) >> circuit
    if circuit.cod != bit ** len(circuit.cod):
        circuit = circuit >> Id(0).tensor(*(
            Discard() if x.name == "qubit" else Id(bit) for x in circuit.cod))

    tk_circ, bits, qubits = Circuit(), [], []

    def remove_ket1(box):
        if not isinstance(box, Ket):
            return box
        x_gates = Id(0).tensor(*(X if x else Id(1) for x in box.bitstring))
        return Ket(*(len(box.bitstring) * (0, ))) >> x_gates

    def prepare_qubits(qubits, left, ket, right):
        renaming = dict()
        start = qubits[-right.count(qubit)] if qubits else tk_circ.n_qubits
        for i in range(start, tk_circ.n_qubits):
            old = Qubit('q', i)
            new = Qubit('q', i + len(ket.cod))
            renaming.update({old: new})
        tk_circ.rename_units(renaming)
        tk_circ.add_blank_wires(len(ket.cod))
        return qubits[:left.count(qubit)]\
            + list(range(start, start + len(ket.cod)))\
            + qubits[-right.count(qubit):]

    def measure_qubits(qubits, bits, left, box, right):
        for j, _ in enumerate(box.dom):
            i_bit, i_qubit = len(tk_circ.bits), qubits[left.count(qubit) + j]
            tk_circ.add_bit(Bit(i_bit))
            tk_circ.Measure(i_qubit, i_bit)
            if isinstance(box, Bra):
                tk_circ.post_select({i_bit: box.bitstring[j]})
            if isinstance(box, Measure):
                bits = bits[:left.count(bit) + j]+ [i_bit]\
                    + bits[-right.count(bit):]
        return bits

    def swap(i, j, unit_factory=Qubit, register='q'):
        old = unit_factory(register, i)
        tmp = unit_factory('tmp', 0)
        new = unit_factory(register, j)
        tk_circ.rename_units({old: tmp})
        tk_circ.rename_units({new: old})
        tk_circ.rename_units({tmp: new})

    def add_gate(qubits, left, box, right):
        offset = left.count(qubit)
        i_qubits = [qubits[offset + j] for j in range(len(box.dom))]
        if isinstance(box, (Rx, Rz)):
            tk_circ.__getattribute__(box.name[:2])(2 * box.phase, *i_qubits)
        elif isinstance(box, CRz):
            tk_circ.__getattribute__(box.name[:3])(2 * box.phase, *i_qubits)
        else:
            tk_circ.__getattribute__(box.name)(*i_qubits)

    circuit = CircuitFunctor(ob=lambda x: x, ar=remove_ket1)(circuit)
    for left, box, right in circuit.layers:
        if isinstance(box, Ket):
            qubits = prepare_qubits(qubits, left, box, right)
        elif isinstance(box, (Measure, Bra)):
            bits = measure_qubits(qubits, bits, left, box, right)
            qubits = qubits[:left.count(qubit)] + qubits[-right.count(qubit):]
        elif isinstance(box, Discard):
            bits = bits[:left.count(bit)] + bits[-right.count(bit):]
            qubits = qubits[:left.count(qubit)] + qubits[-right.count(qubit):]
        elif isinstance(box, Swap):
            if box == Swap(qubit, qubit):
                off = left.count(qubit)
                swap(qubits[off], qubits[off + 1])
            elif box == Swap(bit, bit):
                off = left.count(bit)
                swap(bits[off], bits[off + 1], unit_factory=Bit, register='c')
            elif box in [Swap(qubit, bit), Swap(bit, qubit)]:
                continue
            else:
                raise ValueError(messages.type_err(BitsAndQubits, box.dom))
        elif not box.dom and not box.cod:
            tk_circ.scale(box.array[0])
        else:
            add_gate(qubits, left, box, right)
    return tk_circ


def from_tk(tk_circuit):
    """
    Translates from tket to discopy.
    """
    if not isinstance(tk_circuit, (tk.Circuit, Circuit)):
        raise TypeError(messages.type_err(tk.Circuit, tk_circuit))

    def box_from_tk(tk_gate):
        name = tk_gate.op.type.name
        if name == 'Rx':
            return Rx(tk_gate.op.params[0] / 2)
        if name == 'Rz':
            return Rz(tk_gate.op.params[0] / 2)
        if name == 'CRz':
            return CRz(tk_gate.op.params[0] / 2)
        if name == 'Measure':
            if tk_gate.qubits[0].index[0] in post_selection:
                return Bra(post_selection[tk_gate.qubits[0].index[0]])
            assert tk_gate.bits[0].index[0] == tk_gate.qubits[0].index[0]
            return Measure()
        for gate in gates:
            if name == gate.name:
                return gate
        raise NotImplementedError

    def make_qubits_adjacent(tk_circuit, tk_gate):
        n_qubits, i_0 = tk_circuit.n_qubits, tk_gate.qubits[0].index[0]
        result = Id(n_qubits)
        for i, tk_qubit in enumerate(tk_gate.qubits[1:]):
            if tk_qubit.index[0] == i_0 + i + 1:
                continue  # gate applies to adjacent qubit already
            if tk_qubit.index[0] < i_0 + i + 1:
                for j in range(tk_qubit.index[0], i_0 + i):
                    result = result >> Id(j) @ SWAP @ Id(n_qubits - j - 2)
                if tk_qubit.index[0] <= i_0:
                    i_0 -= 1
            else:
                for j in range(tk_qubit.index[0] - i_0 + i - 1):
                    left = tk_qubit.index[0] - j - 1
                    right = n_qubits - left - 2
                    result = result >> Id(left) @ SWAP @ Id(right)
        return i_0, result
    post_selection, scal = (tk_circuit.post_selection, tk_circuit.scalar)\
        if isinstance(tk_circuit, Circuit) else ({}, 1)
    circuit, n_bits = Ket(*(tk_circuit.n_qubits * [0])), 0
    for tk_gate in tk_circuit.get_commands():
        box = box_from_tk(tk_gate)
        i_0, perm = make_qubits_adjacent(tk_circuit, tk_gate)
        left, right = i_0, len(circuit.cod) - i_0 - len(box.dom)
        if isinstance(box, Measure):
            assert left == n_bits
            layer = Id(bit ** n_bits) @ Id(left - 1) @ box @ Id(right)
            circuit = circuit >> layer
            n_bits += 1
        else:
            layer = Id(bit ** n_bits) @ Id(left) @ box @ Id(right)
            circuit = circuit >> perm >> layer >> perm[::-1]
    for n_qubits, value in reversed(sorted(post_selection.items())):
        circuit = circuit >> Id(bit ** n_bits @ qubit ** n_qubits) @ Bra(value)
    if scal != 1:
        circuit = circuit @ scalar(scal)
    return circuit >> Discard()


def get_counts(circuit, backend, n_shots=2**10, scale=True, post_select=True,
               compilation=None, normalize=True, measure_all=False, seed=None):
    tk_circ = circuit.to_tk()
    if measure_all:
       tk_circ.measure_all()
    if compilation is not None:
       compilation.apply(tk_circ)
    counts = backend.get_counts(tk_circ, n_shots=n_shots, seed=seed)
    if not counts:
       raise RuntimeError
    if normalize:
       counts = probs_from_counts(counts)
    if post_select:
       post_selected = dict()
       for bitstring, count in counts.items():
           if all(bitstring[index] == value
                   for index, value in tk_circ.post_selection.items()):
               post_selected.update({
                   tuple(value for index, value in enumerate(bitstring)
                         if index not in tk_circ.post_selection): count})
       counts = post_selected
    if scale:
        for bitstring in counts:
            counts[bitstring] *= abs(tk_circ.scalar) ** 2
    return counts
