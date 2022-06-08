from discopy.quantum import Circuit
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


def get_valid_states(post_sel: dict, n_wires: int):
    keep_indices = []
    fixed = ['0' if post_sel.get(i, 0) == 0 else '1' for i in range(n_wires)]
    open_wires = set(range(n_wires)) - post_sel.keys()
    permutations = [''.join(s) for s in product('01', repeat=len(open_wires))]
    for perm in permutations:
        new = fixed.copy()
        for i, open in enumerate(open_wires):
            new[open] = perm[i]
        keep_indices.append(int(''.join(new), 2))
    return keep_indices


def extract_ops_from_tk(tk_circ: Circuit, str_map):
    op_list, params_list, wires_list = [], [], []

    for op in tk_circ.__iter__():
        if op.op.type != OpType.Measure:
            op, params, wires = tk_op_to_pennylane(op, str_map)
            op_list.append(op)
            params_list.append([np.pi * p for p in params])
            wires_list.append(wires)

    return op_list, params_list, wires_list


def get_string_repr(circuit: Circuit, post_selection):
    wires = qml.draw(circuit)().split("\n")
    for k, v in post_selection.items():
        wires[k] = wires[k].split("┤")[0] + "┤" + str(v) + ">"

    return "\n".join(wires)


def to_pennylane(circuit: Circuit):
    symbols = circuit.free_symbols
    str_map = {str(s): s for s in symbols}

    tk_circ = circuit.to_tk()
    op_list, params_list, wires_list = extract_ops_from_tk(circuit.to_tk(),
                                                           str_map)

    dev = qml.device('default.qubit', wires=tk_circ.n_qubits, shots=None)

    @qml.qnode(dev, interface="torch")
    def circuit(circ_params):
        for op, params, wires in zip(op_list, circ_params, wires_list):
            op(*params, wires=wires)

        return qml.state()

    def post_selected_circuit(circ_params):
        probs = circuit(circ_params)

        post_selection = tk_circ.post_selection
        open_wires = tk_circ.n_qubits - len(post_selection)
        valid_states = get_valid_states(post_selection, tk_circ.n_qubits)

        post_selected_probs = probs[list(valid_states)]

        return torch.reshape(post_selected_probs, (2,) * open_wires)

    return PennylaneCircuit(post_selected_circuit,
                            params_list,
                            "")


class PennylaneCircuit:
    def __init__(self, circuit, params, string_repr):
        self.circuit = circuit
        self.params = params
        self._contains_sympy = self.contains_sympy()
        self.string_repr = string_repr

    def contains_sympy(self):
        for expr_list in self.params:
            if any(isinstance(expr, sympy.Expr) for
                   expr in expr_list):
                return True
        return False

    def draw(self):
        print(self.string_repr)

    def param_substitution(self, symbols, weights):
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

    def __call__(self, symbols=None, weights=None):
        if self._contains_sympy:
            concrete_params = self.param_substitution(symbols, weights)
            return self.circuit(concrete_params)
        else:
            return self.circuit([torch.cat(p) if len(p) > 0 else p
                                 for p in self.params])
