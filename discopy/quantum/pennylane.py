from discopy.quantum import Circuit
from discopy.quantum.gates import Scalar
from itertools import product
import numpy as np
import pennylane as qml
from pytket import OpType
import sympy
import torch


OP_MAP = {
    OpType.X: qml.PauliX,
    OpType.Y: qml.PauliY,
    OpType.Z: qml.PauliZ,
    OpType.S: qml.S,
    OpType.Sdg: lambda wires: qml.S(wires=wires).inv(),
    OpType.T: qml.T,
    OpType.Tdg: lambda wires: qml.T(wires=wires).inv(),
    OpType.H: qml.Hadamard,
    OpType.Rx: qml.RX,
    OpType.Ry: qml.RY,
    OpType.Rz: qml.RZ,
    OpType.CX: qml.CNOT,
    OpType.CY: qml.CY,
    OpType.CZ: qml.CZ,
    OpType.CRx: qml.CRX,
    OpType.CRy: qml.CRY,
    OpType.CRz: qml.CRZ,
    OpType.CU1: lambda a, wires: qml.ctrl(qml.U1(a, wires=wires[1]),
                                          control=wires[0]),
    OpType.SWAP: qml.SWAP,
    OpType.noop: qml.Identity,
}


def tk_op_to_pennylane(tk_op, str_map):
    """
    Extract the operation, parameters and wires from
    a tket Op, and return the corresponding Pennylane op
    """
    wires = [x.index[0] for x in tk_op.qubits]
    params = tk_op.op.params

    remapped_params = []
    for param in params:
        if isinstance(param, sympy.Expr):
            free_symbols = param.free_symbols
            sym_subs = {f: str_map[str(f)] for f in free_symbols}
            param = param.subs(sym_subs)
        else:
            param = torch.tensor([param])

        remapped_params.append(param)

    return OP_MAP[tk_op.op.type], remapped_params, wires


def extract_ops_from_tk(tk_circ: Circuit, str_map):
    """
    Extract the operations, and corresponding parameters and wires
    from a pytket Circuit. Returns these as lists to use in
    constructing Pennylane circuit.
    """
    op_list, params_list, wires_list = [], [], []

    for op in tk_circ.__iter__():
        if op.op.type != OpType.Measure:
            op, params, wires = tk_op_to_pennylane(op, str_map)
            op_list.append(op)
            params_list.append([np.pi * p for p in params])
            wires_list.append(wires)

    return op_list, params_list, wires_list


def get_post_selection_dict(tk_circ):
    """Return post-selections based on qubit indices."""
    q_post_sels = {}
    for q, c in tk_circ.qubit_to_bit_map.items():
        q_post_sels[q.index[0]] = tk_circ.post_selection[c.index[0]]
    return q_post_sels


def to_pennylane(disco_circuit: Circuit, probabilities=False):
    """
    Return a PennylaneCircuit equivalent to the input DisCoPy
    circuit. `probabilties` determines whether the PennylaneCircuit
    returns states (as in DisCoPy), or probabilties (to be more
    compatible with automatic differentiation in Pennylane)
    """
    symbols = disco_circuit.free_symbols
    str_map = {str(s): s for s in symbols}

    tk_circ = disco_circuit.to_tk()
    op_list, params_list, wires_list = extract_ops_from_tk(tk_circ,
                                                           str_map)

    dev = qml.device('default.qubit', wires=tk_circ.n_qubits, shots=None)
    post_selection = get_post_selection_dict(tk_circ)

    scalar = 1
    for box in disco_circuit.boxes:
        if isinstance(box, Scalar):
            scalar *= box.array

    return PennylaneCircuit(op_list,
                            params_list,
                            wires_list,
                            probabilities,
                            post_selection,
                            scalar,
                            tk_circ.n_qubits,
                            dev)


class PennylaneCircuit:
    """
    Implements a pennylane circuit with post-selection
    """

    def __init__(self, ops, params, wires, probabilities,
                 post_selection, scale, n_qubits, device):
        self.ops = ops
        self.params = params
        self._contains_sympy = self.contains_sympy()
        self.wires = wires
        self.probs = probabilities
        self._post_selection = post_selection
        self.scale = scale
        self.n_qubits = n_qubits
        self.device = device
        self._valid_states = self.get_valid_states()

    @property
    def post_selection(self):
        return self._post_selection

    @post_selection.setter
    def post_selection(self, post_selection_dict):
        self._post_selection = post_selection_dict
        self._valid_states = self.get_valid_states()

    @property
    def valid_states(self):
        return self._valid_states

    def contains_sympy(self):
        """
        Determine if the circuit parameters are
        concrete or contain Sympy symbols
        """
        for expr_list in self.params:
            if any(isinstance(expr, sympy.Expr) for
                   expr in expr_list):
                return True
        return False

    def draw(self, symbols=None, weights=None):
        """
        Prints a string representation of the circuit
        similar to `qml.draw`, but including post-selection
        """
        if self._contains_sympy:
            params = self.param_substitution(symbols, weights)
        else:
            params = [torch.cat(p) if len(p) > 0 else p
                      for p in self.params]

        wires = qml.draw(self.make_circuit())(params).split("\n")
        for k, v in self._post_selection.items():
            wires[k] = wires[k].split("┤")[0] + "┤" + str(v) + ">"

        print("\n".join(wires))

    def get_valid_states(self):
        """
        Determines which of the output states of the circuit are
        compatible with the post-selections
        """
        keep_indices = []
        fixed = ['0' if self._post_selection.get(i, 0) == 0 else
                 '1' for i in range(self.n_qubits)]
        open_wires = set(range(self.n_qubits)) - self._post_selection.keys()
        permutations = [''.join(s) for s in product('01',
                                                    repeat=len(open_wires))]
        for perm in permutations:
            new = fixed.copy()
            for i, open in enumerate(open_wires):
                new[open] = perm[i]
            keep_indices.append(int(''.join(new), 2))
        return keep_indices

    def make_circuit(self):
        """
        Construct the qml.QNode, a circuit that can be used with
        autograd to construct hybrid models
        """
        @qml.qnode(self.device, interface="torch")
        def circuit(circ_params):
            for op, params, wires in zip(self.ops, circ_params, self.wires):
                op(*params, wires=wires)

            if self.probs:
                return qml.probs(wires=range(self.n_qubits))
            else:
                return qml.state()

        return circuit

    def post_selected_circuit(self, params):
        """
        Runs the circuit with the given parameters and returns
        the post-selected output
        """
        states = self.make_circuit()(params)

        open_wires = self.n_qubits - len(self._post_selection)
        post_selected_states = states[list(self._valid_states)]

        if self.probs:
            post_selected_states = \
                post_selected_states / post_selected_states.sum().item()
        else:
            post_selected_states = self.scale * post_selected_states

        return torch.reshape(post_selected_states, (2,) * open_wires)

    def param_substitution(self, symbols, weights):
        """
        Substitutes Sympy symbols for concrete parameters
        """
        concrete_params = []
        for expr_list in self.params:
            concrete_list = []
            for expr in expr_list:
                if isinstance(expr, sympy.Expr):
                    f_expr = sympy.lambdify([symbols], expr)
                    expr = f_expr(weights)
                concrete_list.append(expr)
            concrete_params.append(concrete_list)

        return [torch.cat(p) if len(p) > 0 else p
                for p in concrete_params]

    def eval(self, symbols=None, weights=None):
        """
        Evaluates the circuit. The symbols should be those
        from the original DisCoPy diagram, which will be substituted
        for the concrete parameters in weights.
        """
        if self._contains_sympy:
            concrete_params = self.param_substitution(symbols, weights)
            return self.post_selected_circuit(concrete_params)
        else:
            return self.post_selected_circuit([torch.cat(p) if len(p) > 0
                                               else p for p in self.params])
