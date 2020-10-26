# -*- coding: utf-8 -*-

"""
Implements the translation between discopy and pytket.
"""

import pytket as tk
from pytket.circuit import Bit, Qubit
from pytket.utils import probs_from_counts

from discopy import messages
from discopy.quantum import (
    CircuitFunctor, Id, Bits, Bra, Ket, Swap, scalar as scalar_box,
    bit, qubit, Discard, Measure, GATES, X, Rx, Rz, CRz)


class Circuit(tk.Circuit):
    """
    Extend pytket.Circuit with post selection and scalars.
    """
    @staticmethod
    def upgrade(tk_circuit):
        result = Circuit(tk_circuit.n_qubits, len(tk_circuit.bits))
        for gate in tk_circuit:
            name, inputs = gate.op.type.name, gate.op.params + [
                x.index[0] for x in gate.qubits + gate.bits]
            result.__getattribute__(name)(*inputs)
        return result

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
            self.n_qubits, ", {}".format(len(self.bits)) if self.bits else "")]
        gates = list(map(repr_gate, list(self)))
        post_select = ["post_select({})".format(self.post_selection)]\
            if self.post_selection else []
        scalar = ["scale({})".format(x) for x in [self.scalar] if x != 1]
        return '.'.join(init + gates + post_select + scalar)

    @property
    def n_bits(self):
        """ Number of bits in a circuit. """
        return len(self.bits)

    def rename_units(self, renaming):
        """ Rename units in a circuit. """
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
        """ Scale a circuit by a given number. """
        self.scalar *= number
        return self

    def post_select(self, post_selection):
        """ Post select bits on a a given value. """
        self.post_selection.update(post_selection)
        return self


def to_tk(circuit):
    # bits and qubits are lists of register indices, at layer i we want
    # len(bits) == circuit[:i].cod.count(bit) and same for qubits
    tk_circ, bits, qubits = Circuit(), [], []

    def remove_ket1(box):
        if not isinstance(box, Ket):
            return box
        x_gates = Id(0).tensor(*(X if x else Id(1) for x in box.bitstring))
        return Ket(*(len(box.bitstring) * (0, ))) >> x_gates

    def prepare_qubits(qubits, box, offset):
        renaming = dict()
        start = tk_circ.n_qubits if not qubits else 0\
            if not offset else qubits[offset - 1] + 1
        for i in range(start, tk_circ.n_qubits):
            old = Qubit('q', i)
            new = Qubit('q', i + len(box.cod))
            renaming.update({old: new})
        tk_circ.rename_units(renaming)
        tk_circ.add_blank_wires(len(box.cod))
        return qubits[:offset] + list(range(start, start + len(box.cod)))\
            + [i + len(box.cod) for i in qubits[offset:]]

    def prepare_bits(bits, box, offset):
        renaming = dict()
        start = tk_circ.n_bits if not bits else 0\
            if not offset else bits[offset - 1] + 1
        for i in range(start, tk_circ.n_bits):
            old = Bit(i)
            new = Bit(i + len(box.cod))
            renaming.update({old: new})
        tk_circ.rename_units(renaming)
        for i in range(start, start + len(box.cod)):
            tk_circ.add_bit(Bit(i))
        return bits[:offset] + list(range(start, start + len(box.cod)))\
            + [i + len(box.cod) for i in bits[offset:]]

    def measure_qubits(qubits, bits, box, bit_offset, qubit_offset):
        if isinstance(box, Measure) and box.override_bits:
            for j, _ in enumerate(box.dom[:len(box.dom) // 2]):
                i_bit = bits[bit_offset + j]
                i_qubit = qubits[qubit_offset + j]
                tk_circ.Measure(i_qubit, i_bit)
            return bits, qubits
        for j, _ in enumerate(box.dom):
            i_bit, i_qubit = len(tk_circ.bits), qubits[qubit_offset + j]
            tk_circ.add_bit(Bit(i_bit))
            tk_circ.Measure(i_qubit, i_bit)
            if isinstance(box, Bra):
                tk_circ.post_select({i_bit: box.bitstring[j]})
            if isinstance(box, Measure):
                bits = bits[:bit_offset + j] + [i_bit] + bits[bit_offset + j:]
        if isinstance(box, Bra):
            qubits = qubits[:qubit_offset]\
                + qubits[qubit_offset + len(box.dom):]
        return bits, qubits

    def swap(i, j, unit_factory=Qubit):
        old = unit_factory(i)
        tmp = unit_factory('tmp', 0)
        new = unit_factory(j)
        tk_circ.rename_units({old: tmp})
        tk_circ.rename_units({new: old})
        tk_circ.rename_units({tmp: new})

    def add_gate(qubits, box, offset):
        i_qubits = [qubits[offset + j] for j in range(len(box.dom))]
        if isinstance(box, (Rx, Rz)):
            tk_circ.__getattribute__(box.name[:2])(2 * box.phase, *i_qubits)
        elif isinstance(box, CRz):
            tk_circ.__getattribute__(box.name[:3])(2 * box.phase, *i_qubits)
        elif hasattr(tk_circ, box.name):
            tk_circ.__getattribute__(box.name)(*i_qubits)
        else:
            raise NotImplementedError

    circuit = CircuitFunctor(ob=lambda x: x, ar=remove_ket1)(circuit)
    for left, box, right in circuit.layers:
        if isinstance(box, Ket):
            qubits = prepare_qubits(qubits, box, left.count(qubit))
        elif isinstance(box, Bits):
            if 1 in box.bitstring:
                raise NotImplementedError
            bits = prepare_bits(bits, box, left.count(bit))
        elif isinstance(box, (Measure, Bra)):
            bits, qubits = measure_qubits(
                qubits, bits, box, left.count(bit), left.count(qubit))
        elif isinstance(box, Discard):
            bits = bits[:left.count(bit)]\
                + bits[left.count(bit) + box.dom.count(bit):]
            qubits = qubits[:left.count(qubit)]\
                + qubits[left.count(qubit) + box.dom.count(qubit):]
        elif isinstance(box, Swap):
            if box == Swap(qubit, qubit):
                off = left.count(qubit)
                swap(qubits[off], qubits[off + 1])
            elif box == Swap(bit, bit):
                off = left.count(bit)
                swap(bits[off], bits[off + 1], unit_factory=Bit)
        elif not box.dom and not box.cod:
            tk_circ.scale(box.array[0])
        else:
            add_gate(qubits, box, left.count(qubit))
    return tk_circ


def from_tk(tk_circuit):
    """
    Translates from tket to discopy.
    """
    if not isinstance(tk_circuit, tk.Circuit):
        raise TypeError(messages.type_err(tk.Circuit, tk_circuit))
    if not isinstance(tk_circuit, Circuit):
        tk_circuit = Circuit.upgrade(tk_circuit)
    n_bits = tk_circuit.n_bits - len(tk_circuit.post_selection)
    n_qubits = tk_circuit.n_qubits

    def box_from_tk(tk_gate):
        name = tk_gate.op.type.name
        if name == 'Rx':
            return Rx(tk_gate.op.params[0] / 2)
        if name == 'Rz':
            return Rz(tk_gate.op.params[0] / 2)
        if name == 'CRz':
            return CRz(tk_gate.op.params[0] / 2)
        for gate in GATES:
            if name == gate.name:
                return gate
        raise NotImplementedError

    def make_units_adjacent(tk_gate):
        offset = tk_gate.qubits[0].index[0]
        swaps = Id(qubit ** n_qubits @ bit ** n_bits)
        for i, tk_qubit in enumerate(tk_gate.qubits[1:]):
            source, target = tk_qubit.index[0], offset + i + 1
            if source < target:
                left, right = swaps.cod[:source], swaps.cod[target:]
                swap = Id.swap(
                    swaps.cod[source:source + 1], swaps.cod[source + 1:target])
                if source <= offset:
                    offset -= 1
            elif source > target:
                left, right = swaps.cod[:target], swaps.cod[source + 1:]
                swap = Id.swap(
                    swaps.cod[target: target + 1],
                    swaps.cod[target + 1: source + 1])
            else:  # pragma: no cover
                continue  # units are adjacent already
            swaps = swaps >> Id(left) @ swap @ Id(right)
        return offset, swaps
    circuit, bras = Id(qubit ** n_qubits @ bit ** n_bits), {}
    for tk_gate in tk_circuit.get_commands():
        if tk_gate.op.type.name == "Measure":
            offset = tk_gate.qubits[0].index[0]
            bit_index = tk_gate.bits[0].index[0]
            if bit_index in tk_circuit.post_selection:
                bras[offset] = tk_circuit.post_selection[bit_index]
                continue  # post selection happens at the end
            box = Measure(destructive=False, override_bits=True)
            swaps = Id(circuit.cod[:offset + 1]) @ Id.swap(
               circuit.cod[offset + 1:n_qubits + bit_index],
               circuit.cod[n_qubits:][bit_index: bit_index + 1])\
               @ Id(circuit.cod[n_qubits + bit_index + 1:])
        else:
            box = box_from_tk(tk_gate)
            offset, swaps = make_units_adjacent(tk_gate)
        left, right = swaps.cod[:offset], swaps.cod[offset + len(box.dom):]
        circuit = circuit >> swaps >> Id(left) @ box @ Id(right) >> swaps[::-1]
    circuit = circuit >> Id(0).tensor(*(
        Bra(bras[i]) if i in bras else Id(circuit.cod[i: i + 1])
        for i, _ in enumerate(circuit.cod)))
    if tk_circuit.scalar != 1:
        circuit = circuit @ scalar_box(tk_circuit.scalar)
    return circuit


def get_counts(circuit, backend, n_shots=2**10, scale=True, post_select=True,
               compilation=None, normalize=True, measure_all=False, seed=None):
    """ Runs a circuit on a backend and returns the counts """
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
