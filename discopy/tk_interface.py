# -*- coding: utf-8 -*-

"""
Implements the translation between discopy and pytket.
"""

from discopy import messages
from discopy.cat import Quiver
from discopy.tensor import np, Dim, Tensor
from discopy.circuit import (
    CircuitFunctor, Circuit, Id, Bra, Ket, PRO,
    Rx, Rz, SWAP, CX, H, S, T, X, Y, Z, scalar)

import pytket as tk
from pytket.circuit import UnitID
from pytket.utils import probs_from_counts


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


def to_tk(self):
    def remove_ket1(box):
        if not isinstance(box, Ket):
            return box
        x_gates = Circuit.id(0)
        for bit in box.bitstring:
            x_gates = x_gates @ (X if bit else Circuit.id(1))
        return Ket(*(len(box.bitstring) * (0, ))) >> x_gates

    def swap(tk_circ, i, j):
        old = UnitID('q', i)
        tmp = UnitID('tmp', 0)
        new = UnitID('q', j)
        tk_circ.rename_units({old: tmp})
        tk_circ.rename_units({new: old})
        tk_circ.rename_units({tmp: new})

    def prepare_qubit(tk_circ, left, box, right):
        if len(right) > 0:
            renaming = dict()
            for i in range(len(left), tk_circ.n_qubits):
                old = UnitID('q', i)
                new = UnitID('q', i + len(box.cod))
                renaming.update({old: new})
            tk_circ.rename_units(renaming)
        tk_circ.add_blank_wires(len(box.cod))

    def add_gate(tk_circ, box, off):
        qubits = [off + j for j in range(len(box.dom))]
        if isinstance(box, (Rx, Rz)):
            tk_circ.__getattribute__(box.name[:2])(2 * box.phase, *qubits)
        else:
            tk_circ.__getattribute__(box.name)(*qubits)

    def measure_qubit(tk_circ, left, box, right):
        if len(right) > 0:
            renaming = dict()
            for i, _ in enumerate(box.dom):
                old = UnitID('q', len(left) + i)
                tmp = UnitID('tmp', i)
                renaming.update({old: tmp})
            for i, _ in enumerate(right):
                old = UnitID('q', len(left @ box.dom) + i)
                new = UnitID('q', len(left) + i)
                renaming.update({old: new})
            tk_circ.rename_units(renaming)
            renaming = dict()
            for j, _ in enumerate(box.dom):
                tmp = UnitID('tmp', j)
                new = UnitID('q', len(left @ right) + j)
                renaming.update({tmp: new})
            tk_circ.rename_units(renaming)
        return {len(left @ right) + j: box.bitstring[j]
                for j, _ in enumerate(box.dom)}
    circuit = CircuitFunctor(ob=Quiver(len), ar=Quiver(remove_ket1))(self)
    if circuit.dom != PRO(0):
        circuit = Ket(*(len(circuit.dom) * (0, ))) >> circuit
    tk_circ = TketCircuit()
    for left, box, right in circuit.layers:
        if isinstance(box, Ket):
            prepare_qubit(tk_circ, left, box, right)
        elif isinstance(box, Bra):
            tk_circ.post_selection.update(
                measure_qubit(tk_circ, left, box, right))
        elif box == SWAP:
            swap(tk_circ, len(left), len(left) + 1)
        elif box.dom == box.cod == PRO(0):
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
        name = tk_gate.op.get_type().name
        if name == 'Rx':
            return Rx(tk_gate.op.get_params()[0] / 2)
        if name == 'Rz':
            return Rz(tk_gate.op.get_params()[0] / 2)
        for gate in [SWAP, CX, H, S, T, X, Y, Z]:
            if name == gate.name:
                return gate
        raise NotImplementedError

    def permute(tk_circuit, tk_gate):
        n_qubits, i_0 = tk_circuit.n_qubits, tk_gate.qubits[0].index[0]
        result = Id(n_qubits)
        for i, qubit in enumerate(tk_gate.qubits[1:]):
            if qubit.index[0] == i_0 + i + 1:
                continue  # gate applies to adjacent qubit already
            if qubit.index[0] < i_0 + i + 1:
                for j in range(qubit.index[0], i_0 + i):
                    result = result >> Id(j) @ SWAP @ Id(n_qubits - j - 2)
                if qubit.index[0] <= i_0:
                    i_0 -= 1
            else:
                for j in range(qubit.index[0] - i_0 + i - 1):
                    left = qubit.index[0] - j - 1
                    right = n_qubits - left - 2
                    result = result >> Id(left) @ SWAP @ Id(right)
        return i_0, result
    post_selection, scal = (tk_circuit.post_selection, tk_circuit.scalar)\
        if isinstance(tk_circuit, TketCircuit) else ({}, 1)
    circuit = Id(tk_circuit.n_qubits)
    for tk_gate in tk_circuit.get_commands():
        box = box_from_tk(tk_gate)
        i_0, perm = permute(tk_circuit, tk_gate)
        left, right = i_0, len(circuit.cod) - i_0 - len(box.dom)
        layer = Id(left) @ box @ Id(right)
        circuit = circuit >> perm >> layer >> perm[::-1]
    for qubit, bit in reversed(sorted(post_selection.items())):
        circuit = circuit >> Id(qubit) @ Bra(bit)
    if scal != 1:
        circuit = circuit @ scalar(scal)
    return circuit


def get_counts(self, backend, n_shots=2**10, measure_all=True,
               normalize=True, scale=True, post_select=True, seed=None):
    tk_circ = self.to_tk()
    if measure_all:
        tk_circ.measure_all()
    backend.default_compilation_pass.apply(tk_circ)
    result = backend.get_counts(tk_circ, n_shots=n_shots, seed=seed)
    if not result:  # pragma: no cover
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
            if all(bitstring[qubit] == bit
                    for qubit, bit in post_selection.items()):
                post_selected.update({
                    tuple(bit for qubit, bit in enumerate(bitstring)
                          if qubit not in post_selection): count})
        n_qubits -= len(post_selection.keys())
        counts = post_selected
    array = np.zeros(n_qubits * (2, ))
    for bitstring, count in counts.items():
        array += count * Ket(*bitstring).array
    array = abs(scalar) ** 2 * array
    return Tensor(Dim(1), Dim(*(n_qubits * (2, ))), array)
