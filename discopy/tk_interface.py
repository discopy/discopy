# -*- coding: utf-8 -*-

"""
Implements the translation between discopy and pytket.
"""

from warnings import warn

import pytket as tk
from pytket.circuit import Bit, Qubit
from pytket.utils import probs_from_counts

from discopy import messages
from discopy.cat import Quiver
from discopy.tensor import np, Dim, Tensor
from discopy.quantum import (
    CircuitFunctor, Circuit, Id, Bra, Ket, BitsAndQubits,
    bit, qubit, Discard, Measure, gates, SWAP, X, Rx, Rz, CRz, scalar)


class TketCircuit(tk.Circuit):
    """
    pytket.Circuit with post selection and scalars.

    >>> tk_circ = TketCircuit(post_selection={0: 1}, scalar=2)
    >>> tk_circ.scalar
    2
    >>> tk_circ.post_selection
    {0: 1}
    """
    def __init__(self, post_selection=None, scalar=None):
        self.post_selection = post_selection or {}
        self.scalar = scalar or 1
        super().__init__()

    def __repr__(self):
        return str(self).replace('>', "{}{}>".format(
            ", post_selection={}".format(self.post_selection)
            if self.post_selection else "",
            ", scalar={}".format(self.scalar) if self.scalar != 1 else ""))

    def rename_units(self, renaming):
        """
        >>> circuit = Ket(0)\\
        ...    >> Ket(0, 0) @ Id(1)\\
        ...    >> Id(1) @ Bra(0, 0)\\
        ...    >> Ket(0) @ Id(1)\\
        ...    >> Bra(0, 0)
        >>> tk_circ = to_tk(circuit)
        >>> sorted(tk_circ.post_selection.items())
        [(0, 0), (1, 0), (2, 0), (3, 0)]
        """
        post_selection_renaming = {
            new.index[0]: self.post_selection[old.index[0]]
            for old, new in renaming.items()
            if old.index[0] in self.post_selection}
        for old in renaming.keys():
            if old.index[0] in self.post_selection:
                del self.post_selection[old.index[0]]
        self.post_selection.update(post_selection_renaming)
        super().rename_units(renaming)


def to_tk(circuit):
    def remove_ket1(box):
        if not isinstance(box, Ket):
            return box
        x_gates = Id(0).tensor(*(X if x else Id(1) for x in box.bitstring))
        return Ket(*(len(box.bitstring) * (0, ))) >> x_gates

    def swap(tk_circ, i, j):
        old = Qubit('q', i)
        tmp = Qubit('tmp', 0)
        new = Qubit('q', j)
        tk_circ.rename_units({old: tmp})
        tk_circ.rename_units({new: old})
        tk_circ.rename_units({tmp: new})

    def prepare_qubits(tk_circ, left, box, right):
        if len(right) > 0:
            renaming = dict()
            for i in range(len(left), tk_circ.n_qubits):
                old = Qubit('q', i)
                new = Qubit('q', i + len(box.cod))
                renaming.update({old: new})
            tk_circ.rename_units(renaming)
        tk_circ.add_blank_wires(len(box.cod))

    def add_gate(tk_circ, box, off):
        qubits = [off + j for j in range(len(box.dom))]
        if isinstance(box, (Rx, Rz)):
            tk_circ.__getattribute__(box.name[:2])(2 * box.phase, *qubits)
        elif isinstance(box, CRz):
            tk_circ.__getattribute__(box.name[:3])(2 * box.phase, *qubits)
        else:
            tk_circ.__getattribute__(box.name)(*qubits)

    def move_qubits_right(tk_circ, left, box, right):
        if len(right) > 0:
            renaming = dict()
            for i, _ in enumerate(box.dom):
                old = Qubit('q', len(left) + i)
                tmp = Qubit('tmp', i)
                renaming.update({old: tmp})
            for i, _ in enumerate(right):
                old = Qubit('q', len(left @ box.dom) + i)
                new = Qubit('q', len(left) + i)
                renaming.update({old: new})
            tk_circ.rename_units(renaming)
            renaming = dict()
            for j, _ in enumerate(box.dom):
                tmp = Qubit('tmp', j)
                new = Qubit('q', len(left @ right) + j)
                renaming.update({tmp: new})
            tk_circ.rename_units(renaming)
    if not circuit.dom == qubit ** len(circuit.dom):
        raise ValueError("Circuit should have qubits as domains.")
    if circuit.dom:
        kets = Ket(*(len(circuit.dom) * [0]))
        warn("Pre-composing circuit with {}.".format(kets))
        circuit = kets >> circuit
    if not circuit.cod == bit ** len(circuit.cod):
        discards = Id(0).tensor(*(
            Discard() if x.name == "qubit" else Id(bit) for x in circuit.cod))
        warn("Post-composing circuit with {}.".format(discards))
        circuit = circuit >> discards
    circuit = CircuitFunctor(
        ob=Quiver(lambda x: x), ar=Quiver(remove_ket1))(circuit)
    tk_circ = TketCircuit()
    for left, box, right in circuit.layers:
        if isinstance(box, Ket):
            prepare_qubits(tk_circ, left, box, right)
        elif isinstance(box, (Bra, Measure)):
            move_qubits_right(tk_circ, left, box, right)
            for j, _ in enumerate(box.dom):
                i_bit, i_qubit = len(tk_circ.bits), len(left @ right) + j
                tk_circ.add_bit(Bit(i_bit))
                tk_circ.Measure(i_qubit, i_bit)
                if isinstance(box, Bra):
                    tk_circ.post_selection[i_qubit] = box.bitstring[j]
        elif isinstance(box, Discard):
            move_qubits_right(tk_circ, left, box, right)
        elif box == SWAP:
            swap(tk_circ, len(left), len(left) + 1)
        elif not box.dom and not box.cod:
            tk_circ.scalar *= box.array[0]
        else:
            add_gate(tk_circ, box, len(left))
    return tk_circ


def from_tk(tk_circuit):
    """
    Translates from tket to discopy.
    """
    if not isinstance(tk_circuit, (tk.Circuit, TketCircuit)):
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
        if isinstance(tk_circuit, TketCircuit) else ({}, 1)
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


def get_counts(circuit, backend, n_shots=2**10, measure_all=True,
               normalize=True, scale=True, post_select=True, seed=None):
    tk_circ = circuit.to_tk()
    if measure_all:
        tk_circ.measure_all()
    backend.default_compilation_pass.apply(tk_circ)
    result = backend.get_counts(tk_circ, n_shots=n_shots, seed=seed)
    if not result:
        raise RuntimeError
    return tensor_from_counts(
        result, tk_circ.post_selection, tk_circ.scalar, normalize)


def tensor_from_counts(counts, post_selection=None, scalar=1, normalize=True):
    """
    Parameters
    ----------
    counts : dict
        From bitstrings to counts.
    post_selection : dict, optional
        From qubit indices to bits.
    scalar : complex, optional
        Scale the output using the Born rule.
    normalize : bool, optional
        Whether to normalize the counts.

    Returns
    -------
    tensor : discopy.tensor.Tensor
        Of dimension :code:`n_qubits * (2, )` for :code:`n_qubits` the number
        of post-selected qubits.
    """
    if normalize:
        counts = probs_from_counts(counts)
    n_qubits = len(list(counts.keys()).pop())
    if post_selection:
        post_selected = dict()
        for bitstring, count in counts.items():
            if all(bitstring[qubit_index] == bit
                    for qubit_index, bit in post_selection.items()):
                post_selected.update({
                    tuple(bit for qubit_index, bit in enumerate(bitstring)
                          if qubit_index not in post_selection): count})
        n_qubits -= len(post_selection.keys())
        counts = post_selected
    array = np.zeros(n_qubits * (2, ))
    for bitstring, count in counts.items():
        array += count * Ket(*bitstring).array
    array = abs(scalar) ** 2 * array
    return Tensor(Dim(1), Dim(*(n_qubits * (2, ))), array)
