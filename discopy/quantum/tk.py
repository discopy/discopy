# -*- coding: utf-8 -*-

"""
Implements the translation between discopy and pytket.
"""

from unittest.mock import Mock

import pytket as tk
from pytket.circuit import (Bit, Op, OpType,
                            Qubit)  # pylint: disable=no-name-in-module
from pytket.utils import probs_from_counts

from discopy import messages
from discopy.quantum.circuit import (
    Functor, Id, bit, qubit, Discard, Measure)
from discopy.quantum.gates import (
    ClassicalGate, Controlled, QuantumGate, Bits, Bra, Ket,
    Swap, Scalar, MixedScalar, GATES, X, Rx, Ry, Rz, CRx,
    CRz, format_number)


OPTYPE_MAP = {"H": OpType.H,
              "X": OpType.X,
              "Y": OpType.Y,
              "Z": OpType.Z,
              "S": OpType.S,
              "T": OpType.T,
              "Rx": OpType.Rx,
              "Ry": OpType.Ry,
              "Rz": OpType.Rz,
              "CX": OpType.CX,
              "CZ": OpType.CZ,
              "CRx": OpType.CRx,
              "CRz": OpType.CRz,
              "Swap": OpType.SWAP,
              }


class Circuit(tk.Circuit):
    """
    Extend pytket.Circuit with counts post-processing.
    """
    @staticmethod
    def upgrade(tk_circuit):
        """ Takes a :class:`pytket.Circuit`, returns a :class:`Circuit`. """
        result = Circuit(tk_circuit.n_qubits, len(tk_circuit.bits))
        for gate in tk_circuit:
            name, inputs = gate.op.type.name, gate.op.params + [
                x.index[0] for x in gate.qubits + gate.bits]
            result.__getattribute__(name)(*inputs)
        return result

    def __init__(self, n_qubits=0, n_bits=0,
                 post_selection=None, scalar=None, post_processing=None):
        self.post_selection = post_selection or {}
        self.scalar = scalar or 1
        self.post_processing = post_processing\
            or Id(bit ** (n_bits - len(self.post_selection)))
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
        scalar = ["scale({})".format(format_number(x))
                  for x in [self.scalar] if x != 1]
        post_process = ["post_process({})".format(repr(d))
                        for d in [self.post_processing] if d]
        return '.'.join(init + gates + post_select + scalar + post_process)

    @property
    def n_bits(self):
        """ Number of bits in a circuit. """
        return len(self.bits)

    def add_bit(self, unit, offset=None):
        """ Add a bit, update post_processing. """
        if offset is not None:
            self.post_processing @= Id(bit)
            self.post_processing >>= Id(bit ** offset)\
                @ Id.swap(self.post_processing.cod[offset:-1], bit)
        super().add_bit(unit)

    def rename_units(self, renaming):
        """ Rename units in a circuit. """
        bits_to_rename = [
            old for old in renaming.keys()
            if isinstance(old, Bit) and old.index[0] in self.post_selection]
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
        """ Post-select bits on a a given value. """
        self.post_selection.update(post_selection)
        return self

    def post_process(self, process):
        """ Classical post-processing. """
        self.post_processing >>= process
        return self

    def get_counts(self, *others, backend=None, **params):
        """ Runs a circuit on a backend and returns the counts. """
        n_shots = params.get("n_shots", 2**10)
        scale = params.get("scale", True)
        post_select = params.get("post_select", True)
        compilation = params.get("compilation", None)
        normalize = params.get("normalize", True)
        measure_all = params.get("measure_all", False)
        seed = params.get("seed", None)
        if measure_all:
            for circuit in (self, ) + others:
                circuit.measure_all()
        if compilation is not None:
            for circuit in (self, ) + others:
                compilation.apply(circuit)
        handles = backend.process_circuits(
            (self, ) + others, n_shots=n_shots, seed=seed)
        counts = [backend.get_result(h).get_counts() for h in handles]
        if normalize:
            counts = list(map(probs_from_counts, counts))
        if post_select:
            for i, circuit in enumerate((self, ) + others):
                post_selected = dict()
                for bitstring, count in counts[i].items():
                    if all(bitstring[index] == value
                           for index, value in circuit.post_selection.items()):
                        key = tuple(
                            value for index, value in enumerate(bitstring)
                            if index not in circuit.post_selection)
                        post_selected.update({key: count})
                counts[i] = post_selected
        if scale:
            for i, circuit in enumerate((self, ) + others):
                for bitstring in counts[i]:
                    counts[i][bitstring] *= circuit.scalar
        return counts


def to_tk(circuit):
    """
    Takes a :class:`discopy.quantum.Circuit`, returns a :class:`Circuit`.
    """
    # bits and qubits are lists of register indices, at layer i we want
    # len(bits) == circuit[:i].cod.count(bit) and same for qubits
    tk_circ, bits, qubits = Circuit(), [], []
    circuit = circuit.init_and_discard()

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
            tk_circ.add_bit(Bit(i), offset=offset + i - start)
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
            offset = len(bits) if isinstance(box, Measure) else None
            tk_circ.add_bit(Bit(i_bit), offset=offset)
            tk_circ.Measure(i_qubit, i_bit)
            if isinstance(box, Bra):
                tk_circ.post_select({i_bit: box.bitstring[j]})
            if isinstance(box, Measure):
                bits = bits[:bit_offset + j] + [i_bit] + bits[bit_offset + j:]
        if isinstance(box, Bra)\
                or isinstance(box, Measure) and box.destructive:
            qubits = qubits[:qubit_offset]\
                + qubits[qubit_offset + len(box.dom):]
        return bits, qubits

    def swap(i, j, unit_factory=Qubit):
        old, tmp, new =\
            unit_factory(i), unit_factory('tmp', 0), unit_factory(j)
        tk_circ.rename_units({old: tmp})
        tk_circ.rename_units({new: old})
        tk_circ.rename_units({tmp: new})

    def add_gate(qubits, box, offset):
        i_qubits = [qubits[offset + j] for j in range(len(box.dom))]

        if isinstance(box, (Rx, Ry, Rz)):
            op = Op.create(OPTYPE_MAP[box.name[:2]], 2 * box.phase)
        elif isinstance(box, Controlled):
            i_qubits = []
            idx = offset if box.distance > 0 else offset - box.distance
            curr_box = box
            while isinstance(curr_box, Controlled):
                i_qubits.append(qubits[idx])
                idx += curr_box.distance
                curr_box = curr_box.controlled
            i_qubits.append(qubits[idx])

            name = box.name.split('(')[0]
            if '(' not in box.name:
                # CX, CZ, CCX
                op = Op.create(OPTYPE_MAP[name])
            elif name in ('CRx', 'CRz'):
                op = Op.create(OPTYPE_MAP[name], 2 * box.phase)
        elif box.name in OPTYPE_MAP:
            op = Op.create(OPTYPE_MAP[box.name])
        else:
            raise NotImplementedError

        if box.is_dagger:
            op = op.dagger

        tk_circ.add_gate(op, i_qubits)

    circuit = Functor(ob=lambda x: x, ar=remove_ket1)(circuit)
    for left, box, _ in circuit.layers:
        if isinstance(box, Ket):
            qubits = prepare_qubits(qubits, box, left.count(qubit))
        elif isinstance(box, Bits) and not box.is_dagger:
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
                if tk_circ.post_processing:
                    right = Id(tk_circ.post_processing.cod[off + 2:])
                    tk_circ.post_process(
                        Id(bit ** off) @ Swap(bit, bit) @ right)
                else:
                    swap(bits[off], bits[off + 1], unit_factory=Bit)
            else:  # pragma: no cover
                continue  # bits and qubits live in different registers.
        elif isinstance(box, Scalar):
            tk_circ.scale(
                box.array if box.is_mixed else abs(box.array) ** 2)
        elif isinstance(box, ClassicalGate)\
                or isinstance(box, Bits) and box.is_dagger:
            off = left.count(bit)
            right = Id(tk_circ.post_processing.cod[off + len(box.dom):])
            tk_circ.post_process(Id(bit ** off) @ box @ right)
        elif isinstance(box, QuantumGate):
            add_gate(qubits, box, left.count(qubit))
        else:  # pragma: no cover
            raise NotImplementedError
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
        if name == 'CRx':
            return CRx(tk_gate.op.params[0] / 2)
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
    circuit = Id(0).tensor(*(n_qubits * [Ket(0)] + n_bits * [Bits(0)]))
    bras = {}
    for tk_gate in tk_circuit.get_commands():
        if tk_gate.op.type.name == "Measure":
            offset = tk_gate.qubits[0].index[0]
            bit_index = tk_gate.bits[0].index[0]
            if bit_index in tk_circuit.post_selection:
                bras[offset] = tk_circuit.post_selection[bit_index]
                continue  # post selection happens at the end
            box = Measure(destructive=False, override_bits=True)
            swaps = Id(circuit.cod[:offset + 1])
            swaps = swaps @ Id.swap(
                circuit.cod[offset + 1:n_qubits + bit_index],
                circuit.cod[n_qubits:][bit_index: bit_index + 1])\
                @ Id(circuit.cod[n_qubits + bit_index + 1:])
        else:
            box = box_from_tk(tk_gate)
            offset, swaps = make_units_adjacent(tk_gate)
        left, right = swaps.cod[:offset], swaps.cod[offset + len(box.dom):]
        circuit = circuit >> swaps >> Id(left) @ box @ Id(right) >> swaps[::-1]
    circuit = circuit >> Id(0).tensor(*(
        Bra(bras[i]) if i in bras
        else Discard() if x.name == 'qubit' else Id(bit)
        for i, x in enumerate(circuit.cod)))
    if tk_circuit.scalar != 1:
        circuit = circuit @ MixedScalar(tk_circuit.scalar)
    return circuit >> tk_circuit.post_processing


def mockBackend(*counts):
    """ Takes a list of counts, returns a mock backend that outputs them. """
    def get_result(i):
        result = Mock()
        result.get_counts.return_value = counts[i]
        return result
    mock = Mock()
    mock.process_circuits.return_value = list(range(len(counts)))
    mock.get_result = get_result
    return mock
