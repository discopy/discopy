"""
Implements a conversion from quantum DisCoPy circuits to
PennyLane circuits.

If `probabilities` is set to False, the output states of the PennyLane
circuit will be exactly equivalent to those of the DisCoPy circuit
(for the same parameters).

If `probabilities` is set to True, the output states of the PennyLane
circuit will be the probabilities of the output states, equivalent
to appending :class:`discopy.quantum.circuit.Measure` to all the
open wires in the DisCoPy circuit.

Once a :class:`PennyLaneCircuit` has been constructed, it
can be evaluated with :func:`.eval()`. If the circuit contains only
concrete parameters (i.e. no symbolic parameters), no arguments
should be passed to `eval()`. If the circuit contains symbolic
parameters, a list of the symbolic parameters and a list of their
associated weights should be passed to `eval()` as `symbols=` and
`weights=`.
"""

from discopy.quantum import Circuit
from discopy.quantum.gates import Scalar
from itertools import product
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


def tk_op_to_pennylane(tk_op):
    """
    Extract the operation, parameters and wires from
    a pytket :class:`Op`, and return the corresponding PennyLane operation.

    Parameters
    ----------
    tk_op : :class:`pytket.circuit.Op`
        The pytket :class:`Op` to convert.
    str_map : dict[str, :class:`sympy.core.symbol.Symbol`]
        A mapping from strings to SymPy symbols (necessary as
        `circ.to_tk()` does not copy symbol references).

    Returns
    -------
    :class:`qml.operation.Operation`
        The PennyLane operation equivalent to the input pytket Op.
    list of (:class:`torch.FloatTensor` or :class:`sympy.core.symbol.Symbol`)
        The parameters of the operation.
    list of int
        The wires/qubits to apply the operation to.
    """
    wires = [x.index[0] for x in tk_op.qubits]
    params = tk_op.op.params

    remapped_params = []
    for param in params:
        # scale rotation from [0, 2) to [0, 1), (rescale to [0, 2pi) later)
        param /= 2
        if not isinstance(param, sympy.Expr):
            param = torch.tensor(param)

        remapped_params.append(param)

    return OP_MAP[tk_op.op.type], remapped_params, wires


def extract_ops_from_tk(tk_circ):
    """
    Extract the operations, and corresponding parameters and wires,
    from a pytket Circuit. Return these as lists to use in
    constructing PennyLane circuit.

    Parameters
    ----------
    tk_circ : :class:`discopy.quantum.tk.Circuit`
        The pytket circuit to extract the operations from.
    str_map : dict of str: :class:`sympy.core.symbol.Symbol`
        A mapping from strings to SymPy symbols.

    Returns
    -------
    list of :class:`qml.operation.Operation`
        The PennyLane operations extracted from the pytket circuit.
    list of list of (:class:`torch.FloatTensor` or
                     :class:`sympy.core.symbol.Symbol`)
        The corresponding parameters of the operations.
    list of list of int
        The corresponding wires of the operations.
    """
    op_list, params_list, wires_list = [], [], []

    for op in tk_circ.__iter__():
        if op.op.type != OpType.Measure:
            op, params, wires = tk_op_to_pennylane(op)
            op_list.append(op)
            params_list.append(params)
            wires_list.append(wires)

    return op_list, params_list, wires_list


def get_post_selection_dict(tk_circ):
    """
    Return post-selections based on qubit indices.

    Parameters
    ----------
    tk_circ : :class:`discopy.quantum.tk.Circuit`
        The pytket circuit to extract the post-selections from.

    Returns
    -------
    dict of int: int
        A mapping from qubit indices to pytket classical indices.
    """
    q_post_sels = {}
    for q, c in tk_circ.qubit_to_bit_map.items():
        q_post_sels[q.index[0]] = tk_circ.post_selection[c.index[0]]
    return q_post_sels


def to_pennylane(disco_circuit: Circuit, probabilities=False,
                 backend_config=None, diff_method="best"):
    """
    Return a PennyLaneCircuit equivalent to the input DisCoPy
    circuit. `probabilties` determines whether the PennyLaneCircuit
    returns states (as in DisCoPy), or probabilties (to be more
    compatible with automatic differentiation in PennyLane).

    Parameters
    ----------
    disco_circuit : :class:`discopy.quantum.circuit.Circuit`
        The DisCoPy circuit to convert to PennyLane.
    probabilities : bool, default: False
        Determines whether the PennyLane
        circuit outputs states or un-normalized probabilities.
        Probabilities can be used with more PennyLane backpropagation
        methods.

    Returns
    -------
    :class:`PennyLaneCircuit`
        The PennyLane circuit equivalent to the input DisCoPy circuit.
    """
    if disco_circuit.is_mixed:
        raise ValueError('Only pure quantum circuits are currently '
                         'supported.')

    tk_circ = disco_circuit.to_tk()
    op_list, params_list, wires_list = extract_ops_from_tk(tk_circ)

    post_selection = get_post_selection_dict(tk_circ)

    scalar = 1
    for box in disco_circuit.boxes:
        if isinstance(box, Scalar):
            scalar *= box.array

    return PennyLaneCircuit(op_list,
                            params_list,
                            wires_list,
                            probabilities,
                            post_selection,
                            scalar,
                            tk_circ.n_qubits,
                            backend_config,
                            diff_method)


STATE_BACKENDS = ['default.qubit', 'lightning.qubit', 'qiskit.aer']
STATE_DEVICES = ['aer_simulator_statevector', 'statevector_simulator']


class PennyLaneCircuit:
    """
    Implement a pennylane circuit with post-selection.
    """
    def __init__(self, ops, params, wires, probabilities,
                 post_selection, scale, n_qubits, backend_config,
                 diff_method):
        self._ops = ops
        self._params = params
        self._wires = wires
        self._probabilities = probabilities
        self._post_selection = post_selection
        self._scale = scale
        self._n_qubits = n_qubits
        self._backend_config = backend_config
        self.diff_method = diff_method

        self._contains_sympy = self.contains_sympy()
        if self._contains_sympy:
            self._concrete_params = None
        else:
            self._concrete_params = params
        self.initialise_device_and_circuit()
        self._valid_states = self.get_valid_states()

    def get_device(self, backend_config):
        """
        Return a PennyLane device with the specified backend
        configuration.
        """
        if backend_config is None:
            backend = 'default.qubit'
            backend_config = {}
        else:
            backend = backend_config.pop('backend')

        if backend == 'honeywell.hqs':
            try:
                backend_config['machine'] = backend_config.pop('device')
            except KeyError:
                raise ValueError('When using the honeywell.hqs provider, '
                                 'a device must be specified.')
        elif 'device' in backend_config:
            backend_config['backend'] = backend_config.pop('device')

        if not self._probabilities:
            if backend not in STATE_BACKENDS:
                raise ValueError(f'The {backend} backend is not '
                                 'compatible with state outputs.')
            elif ('backend' in backend_config
                  and backend_config['backend'] not in STATE_DEVICES):
                raise ValueError(f'The {backend_config["backend"]} '
                                 'device is not compatible with state '
                                 'outputs.')

        return qml.device(backend, wires=self._n_qubits, **backend_config)

    def initialise_device_and_circuit(self):
        """
        Initialise the PennyLane device and circuit when instantiating the
        PennyLaneCirucit, or loading from disk.
        """
        self._device = self.get_device(None if self._backend_config is None
                                       else {**self._backend_config})
        self._circuit = self.make_circuit()

    def contains_sympy(self):
        """
        Determine if the circuit parameters are
        concrete or contain SymPy symbols.

        Returns
        -------
        bool
            Whether the circuit parameters contain SymPy symbols.
        """
        return any(isinstance(expr, sympy.Expr) for expr_list in
                   self._params for expr in expr_list)

    def initialise_concrete_params(self, symbols, weights):
        """
        Given concrete values for each of the SymPy symbols, substitute
        the symbols for the values to obtain concrete parameters, via
        the `param_substitution` method.
        """
        if self._contains_sympy:
            self._concrete_params = self.param_substitution(symbols, weights)

    def draw(self):
        """
        Print a string representation of the circuit
        similar to `qml.draw`, but including post-selection.

        Parameters
        ----------
        symbols : list of :class:`sympy.core.symbol.Symbol`, default: None
            The symbols from the original DisCoPy circuit.
        weights : list of :class:`torch.FloatTensor`, default: None
            The weights to substitute for the symbols.
        """
        if self._concrete_params is None:
            raise ValueError('Cannot draw circuit with symbolic parameters. '
                             'Initialise concrete parameters first.')

        wires = (qml.draw(self._circuit)
                 (self._concrete_params).split("\n"))
        for k, v in self._post_selection.items():
            wires[k] = wires[k].split("┤")[0] + "┤" + str(v) + ">"

        print("\n".join(wires))

    def get_valid_states(self):
        """
        Determine which of the output states of the circuit are
        compatible with the post-selections.

        Returns
        -------
        list of int
            The indices of the circuit output that are
            compatible with the post-selections.
        """
        keep_indices = []
        fixed = ['0' if self._post_selection.get(i, 0) == 0 else
                 '1' for i in range(self._n_qubits)]
        open_wires = set(range(self._n_qubits)) - self._post_selection.keys()
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
        Construct the :class:`qml.Qnode`, a circuit that can be used with
        autograd to construct hybrid models.

        Returns
        -------
        :class:`qml.Qnode`
            A Pennylane circuit without post-selection.
        """
        @qml.qnode(self._device, interface="torch",
                   diff_method=self.diff_method)
        def circuit(circ_params):
            for op, params, wires in zip(self._ops, circ_params, self._wires):
                op(*[2 * torch.pi * p for p in params], wires=wires)

            if self._probabilities:
                return qml.probs(wires=range(self._n_qubits))
            else:
                return qml.state()

        return circuit

    def post_selected_circuit(self, params):
        """
        Run the circuit with the given parameters and return
        the post-selected output.

        Parameters
        ----------
        params : :class:`torch.FloatTensor`
            The concrete parameters for the gates in the circuit.

        Returns
        -------
        :class:`torch.Tensor`
            The post-selected output of the circuit.
        """
        states = self._circuit(params)

        open_wires = self._n_qubits - len(self._post_selection)
        post_selected_states = states[self._valid_states]
        post_selected_states *= (self._scale ** 2 if self._probabilities
                                 else self._scale)

        if post_selected_states.shape[0] == 1:
            return post_selected_states
        else:
            return torch.reshape(post_selected_states, (2,) * open_wires)

    def param_substitution(self, symbols, weights):
        """
        Substitute symbolic parameters (SymPy symbols) with floats.

        Parameters
        ----------
        symbols : list of :class:`sympy.core.symbol.Symbol`
            The symbols from the original DisCoPy circuit.
        weights : list of :class:`torch.FloatTensor`
            The weights to substitute for the symbols.

        Returns
        -------
        :class:`torch.FloatTensor`
            The concrete (non-symbolic) parameters for the
            circuit.
        """
        concrete_params = []
        for expr_list in self._params:
            concrete_list = []
            for expr in expr_list:
                if isinstance(expr, sympy.Expr):
                    f_expr = sympy.lambdify([symbols], expr)
                    expr = f_expr(weights)
                concrete_list.append(expr)
            concrete_params.append(concrete_list)

        return concrete_params

    def eval(self):
        """
        Evaluate the circuit. The symbols should be those
        from the original DisCoPy diagram, which will be substituted
        for the concrete parameters in weights.

        Parameters
        ----------
        symbols : list of :class:`sympy.core.symbol.Symbol`, default: None
            The symbols from the original DisCoPy circuit.
        weights : list of :class:`torch.FloatTensor`, default: None
            The weights to substitute for the symbols.

        Returns
        -------
        :class:`torch.Tensor`
            The post-selected output of the circuit.
        """
        if self._concrete_params is None:
            raise ValueError('Initialise concrete parameters first.')

        return self.post_selected_circuit(self._concrete_params)
