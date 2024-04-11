# -*- coding: utf-8 -*-

"""
The category of classical-quantum circuits with digits and qudits as objects.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Digit
    Qudit
    Ty
    Circuit
    Box
    Sum
    Swap
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        index2bitstring
        bitstring2index

Examples
--------
>>> from discopy.quantum.gates import (
...     Ket, CX, H, X, Rz, sqrt, Controlled, Measure, Discard)
>>> circuit = Ket(0, 0) >> CX >> Controlled(Rz(0.25)) >> Measure() @ Discard()
>>> circuit.draw(
...     figsize=(3, 6),
...     path='docs/_static/quantum/circuit-example.png')

.. image:: /_static/quantum/circuit-example.png
    :align: center

>>> from discopy.grammar import pregroup
>>> s, n = pregroup.Ty('s'), pregroup.Ty('n')
>>> Alice = pregroup.Word('Alice', n)
>>> loves = pregroup.Word('loves', n.r @ s @ n.l)
>>> Bob = pregroup.Word('Bob', n)
>>> grammar = pregroup.Cup(n, n.r) @ s @ pregroup.Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: Ty(), n: qubit}
>>> ar = {Alice: Ket(0),
...       loves: CX << sqrt(2) @ H @ X << Ket(0, 0),
...       Bob: Ket(1)}
>>> F = pregroup.Functor(ob, ar, cod=Category(Ty, Circuit))
>>> assert abs(F(sentence).eval().array) ** 2

>>> from discopy.drawing import Equation
>>> Equation(
...     sentence.to_drawing(), F(sentence), symbol='$\\\\mapsto$').draw(
...         figsize=(6, 3), nodesize=.5,
...         path='docs/_static/quantum/functor-example.png')

.. image:: /_static/quantum/functor-example.png
    :align: center
"""

from __future__ import annotations

from collections.abc import Mapping

from discopy import messages, tensor, frobenius
from discopy.cat import factory, Category
from discopy.matrix import backend
from discopy.tensor import Dim, Tensor
from discopy.utils import factory_name, assert_isinstance


class Ob(frobenius.Ob):
    """
    A circuit object is an information unit with some dimension ``dim > 1``.

    Parameters:
        name : The name of the object, e.g. ``"bit"`` or ``"qubit"``.
        dim : The dimension of the object, e.g. ``2`` for bits and qubits.

    Note
    ----
    This class can only be instantiated via its subclasses :class:`Digit` and
    :class:`Qudit`, but feel free to open a pull-request if you discover a
    third kind of information unit.
    """
    def __init__(self, name: str, dim=2, z=0):
        assert_isinstance(dim, int)
        assert_isinstance(self, (Digit, Qudit))
        if dim < 2:
            raise ValueError
        self.dim = dim
        super().__init__(name, z)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.dim})"

    @classmethod
    def from_tree(cls, tree: dict) -> Ob:
        dim, z = tree['dim'], tree.get('z', 0)
        return cls(dim=dim, z=z)

    def to_tree(self) -> dict:
        return dict(dim=self.dim, **super().to_tree())


class Digit(Ob):
    """
    A digit is a classical unit of information.

    Parameters:
        dim : The dimension of the digit, e.g. ``2`` for bits.

    Examples
    --------
    >>> assert bit.inside == (Digit(2),)
    """
    def __init__(self, dim: int, z=0):
        name = "bit" if dim == 2 else f"Digit({dim})"
        super().__init__(name, dim)


class Qudit(Ob):
    """
    A qudit is a quantum unit of information, i.e. a quantum digit.

    Parameters:
        dim : The dimension of the qudit, e.g. ``2`` for qubits.

    Examples
    --------
    >>> assert qubit.inside == (Qudit(2),)
    """
    def __init__(self, dim, z=0):
        name = "qubit" if dim == 2 else f"Qudit({dim})"
        super().__init__(name, dim)

    def __setstate__(self, state):
        if "_dim" in state:
            state["dim"] = state["_dim"]
            del state["_dim"]
        super().__setstate__(state)


@factory
class Ty(frobenius.Ty):
    """
    A circuit type is a frobenius type with :class:`Digit` and :class:`Qudit`
    objects inside.

    Parameters:
        inside (Digit | Qudit) : The digits and qudits inside the type.

    Examples
    --------
    >>> assert bit == Ty(Digit(2))
    >>> assert qubit == Ty(Qudit(2))
    >>> assert bit @ qubit != qubit @ bit

    You can construct :code:`n` qubits by taking powers of :code:`qubit`:

    >>> print(bit ** 2 @ qubit ** 3)
    bit @ bit @ qubit @ qubit @ qubit
    """
    ob_factory = Ob


@factory
class Circuit(tensor.Diagram[complex]):
    """
    A circuit is a tensor diagram with bits and qubits as ``dom`` and ``cod``.

    Parameters:
        inside (tuple[Layer, ...]) : The layers inside the circuit diagram.
        dom (quantum.circuit.Ty) : The domain of the circuit diagram.
        cod (quantum.circuit.Ty) : The codomain of the circuit diagram.
    """
    ty_factory = Ty

    @classmethod
    def id(cls, dom: int | Ty = None):
        """
        The identity circuit on a given domain.

        Parameters:
            dom : The domain (and codomain) of the identity,
                  or ``qubit ** dom`` if ``dom`` is an ``int``.
        """
        dom = qubit ** dom if isinstance(dom, int) else dom
        return tensor.Diagram.id.__func__(Circuit, dom)

    @property
    def is_mixed(self):
        """
        Whether the circuit is mixed, i.e. it contains both bits and qubits
        or it discards qubits.

        Mixed circuits can be evaluated only by a
        :class:`ChannelFunctor` not a :class:`discopy.tensor.Functor`.
        """
        both_bits_and_qubits = self.dom.count(bit) and self.dom.count(qubit)\
            or any(layer.cod.count(bit) and layer.cod.count(qubit)
                   for layer in self.inside)
        return both_bits_and_qubits or any(box.is_mixed for box in self.boxes)

    def init_and_discard(self):
        """ Returns a circuit with empty domain and only bits as codomain. """
        from discopy.quantum.gates import Bits, Ket, Discard
        circuit = self
        if circuit.dom:
            init = Id().tensor(*(
                Bits(0) if x.name == "bit" else Ket(0) for x in circuit.dom))
            circuit = init >> circuit
        if circuit.cod != bit ** len(circuit.cod):
            discards = Id().tensor(*(
                Discard() if x.name == "qubit"
                else Id(bit) for x in circuit.cod))
            circuit = circuit >> discards
        return circuit

    def eval(self, *others, backend=None, mixed=False,
             contractor=None, **params):
        """
        Evaluate a circuit on a backend, or simulate it with numpy.

        Parameters
        ----------
        others : :class:`discopy.quantum.circuit.Circuit`
            Other circuits to process in batch.
        backend : pytket.Backend, optional
            Backend on which to run the circuit, if none then we apply
            :class:`discopy.tensor.Functor` or :class:`ChannelFunctor` instead.
        mixed : bool, optional
            Whether to apply :class:`discopy.tensor.Functor`
            or :class:`ChannelFunctor`.
        contractor : callable, optional
            Use :class:`tensornetwork` contraction
            instead of discopy's basic eval feature.
        params : kwargs, optional
            Get passed to Circuit.get_counts.

        Returns
        -------
        tensor : Tensor[float]
            If :code:`backend is not None`.
        tensor : Tensor[complex]
            If :code:`mixed=False`.
        channel : :class:`Channel`
            Otherwise.

        Examples
        --------
        We can evaluate a pure circuit (i.e. with :code:`not circuit.is_mixed`)
        as a unitary :class:`discopy.tensor.Tensor` or as a :class:`Channel`:

        >>> from discopy.quantum import *

        >>> H.eval().round(2)  # doctest: +ELLIPSIS
        Tensor[complex]([0.71+0.j, ..., -0.71+0.j], dom=Dim(2), cod=Dim(2))
        >>> H.eval(mixed=True).round(1)  # doctest: +ELLIPSIS
        Channel([0.5+0.j, ..., 0.5+0.j], dom=Q(Dim(2)), cod=Q(Dim(2)))

        We can evaluate a mixed circuit as a :class:`Channel`:

        >>> from discopy.quantum import Channel
        >>> assert Measure().eval()\\
        ...     == Channel(dom=Q(Dim(2)), cod=C(Dim(2)),
        ...              array=[1, 0, 0, 0, 0, 0, 0, 1])
        >>> circuit = Bits(1, 0) @ Ket(0) >> Discard(bit ** 2 @ qubit)
        >>> assert circuit.eval() == Channel(dom=CQ(), cod=CQ(), array=[1])

        We can execute any circuit on a `pytket.Backend` and get a
        :class:`discopy.tensor.Tensor` of real-valued probabilities.

        >>> circuit = Ket(0, 0) >> sqrt(2) @ H @ X >> CX >> Measure() @ Bra(0)
        >>> from discopy.quantum.tk import mockBackend
        >>> backend = mockBackend({(0, 1): 512, (1, 0): 512})
        >>> assert circuit.eval(backend=backend, n_shots=2**10).round()\\
        ...     == Tensor[float](dom=Dim(1), cod=Dim(2), array=[0., 1.])

        Note
        ----
        Any extra parameter is passed to :meth:`Circuit.get_counts`.
        For instance, to evaluate a unitary circuit (i.e. with no measurements)
        on a ``pytket.Backend`` one should set ``measure_all=True``.
        """
        from discopy.quantum import channel
        if contractor is not None:
            array = contractor(*self.to_tn(mixed=mixed)).tensor
            if self.is_mixed or mixed:
                f = channel.Functor({}, {}, dom=Category(Ty, Circuit))
                return channel.Channel(array, f(self.dom), f(self.cod))
            f = tensor.Functor(
                lambda x: x.inside[0].dim, {},
                dtype=complex, dom=Category(Ty, Circuit))
            return Tensor[complex](array, f(self.dom), f(self.cod))

        from discopy.quantum import channel
        if backend is None:
            if others:
                return [circuit.eval(mixed=mixed, **params)
                        for circuit in (self, ) + others]
            if mixed or self.is_mixed:
                return channel.Functor(
                    {}, {}, dom=Category(Ty, Circuit), dtype=complex)(self)
            return tensor.Functor(
                lambda x: x.inside[0].dim,
                lambda f: f.array,
                dom=Category(Ty, Circuit),
                dtype=complex)(self)
        circuits = [circuit.to_tk() for circuit in (self, ) + others]
        results, counts = [], circuits[0].get_counts(
            *circuits[1:], backend=backend, **params)
        for i, circuit in enumerate(circuits):
            n_bits = len(circuit.post_processing.dom)
            result = Tensor[float].zero(Dim(1), Dim(*(n_bits * (2, ))))
            for bitstring, count in counts[i].items():
                result.array[bitstring] = count
            if circuit.post_processing:
                result = result >> circuit.post_processing.eval().cast(float)
            results.append(result)
        return results if len(results) > 1 else results[0]

    def get_counts(self, *others, backend=None, **params):
        """
        Get counts from a backend, or simulate them with numpy.

        Parameters
        ----------
        others : :class:`discopy.quantum.circuit.Circuit`
            Other circuits to process in batch.
        backend : pytket.Backend, optional
            Backend on which to run the circuit, if none then `numpy`.
        n_shots : int, optional
            Number of shots, default is :code:`2**10`.
        measure_all : bool, optional
            Whether to measure all qubits, default is :code:`False`.
        normalize : bool, optional
            Whether to normalize the counts, default is :code:`True`.
        post_select : bool, optional
            Whether to perform post-selection, default is :code:`True`.
        scale : bool, optional
            Whether to scale the output, default is :code:`True`.
        seed : int, optional
            Seed to feed the backend, default is :code:`None`.
        compilation : callable, optional
            Compilation function to apply before getting counts.

        Returns
        -------
        counts : dict
            From bitstrings to counts.

        Examples
        --------
        >>> from discopy.quantum import *
        >>> circuit = H @ X >> CX >> Measure(2)
        >>> from discopy.quantum.tk import mockBackend
        >>> backend = mockBackend({(0, 1): 512, (1, 0): 512})
        >>> circuit.get_counts(backend=backend, n_shots=2**10)
        {(0, 1): 0.5, (1, 0): 0.5}
        """
        if backend is None:
            if others:
                return [circuit.get_counts(**params)
                        for circuit in (self, ) + others]
            result, counts = self.init_and_discard().eval(mixed=True), dict()
            for i in range(2 ** len(result.cod.classical)):
                bits = index2bitstring(i, len(result.cod.classical))
                if result.array[bits]:
                    counts[bits] = result.array[bits].real
            return counts
        counts = self.to_tk().get_counts(
            *(other.to_tk() for other in others), backend=backend, **params)
        return counts if len(counts) > 1 else counts[0]

    def measure(self, mixed=False):
        """
        Measure a circuit on the computational basis using :code:`numpy`.

        Parameters
        ----------
        mixed : Whether to apply a :class:`tensor.Functor`
                or a :class:`channel.Functor`.

        Returns
        -------
        array : numpy.ndarray
        """
        from discopy.quantum.gates import Bra, Ket
        if mixed or self.is_mixed:
            return self.init_and_discard().eval(mixed=True).array.real
        state = (Ket(*(len(self.dom) * [0])) >> self).eval()
        effects = [Bra(*index2bitstring(j, len(self.cod))).eval()
                   for j in range(2 ** len(self.cod))]
        with backend() as np:
            array = np.zeros(len(self.cod) * (2, )) + 0j
            for effect in effects:
                array +=\
                    effect.array * np.absolute((state >> effect).array) ** 2
        return array

    def to_tn(self, mixed=False):
        """
        Send a diagram to a mixed :code:`tensornetwork`.

        Parameters
        ----------
        mixed : bool, default: False
            Whether to perform mixed (also known as density matrix) evaluation
            of the circuit.

        Returns
        -------
        nodes : :class:`tensornetwork.Node`
            Nodes of the network.

        output_edge_order : list of :class:`tensornetwork.Edge`
            Output edges of the network.
        """
        if not mixed and not self.is_mixed:
            return super().to_tn(dtype=complex)

        import tensornetwork as tn
        from discopy.quantum.gates import (
            ClassicalGate, Copy, Match, Discard, Measure, Encode, SWAP)
        for box in self.boxes + [self]:
            if set(box.dom @ box.cod) - set(bit @ qubit):
                raise ValueError(
                    "Only circuits with qubits and bits are supported.")

        # try to decompose some gates
        diag = Id(self.dom)
        last_i = 0
        for i, box in enumerate(self.boxes):
            if hasattr(box, '_decompose'):
                decomp = box._decompose()
                diag >>= self[last_i:i]
                left, _, right = self.inside[i]
                diag >>= Id(left) @ decomp @ Id(right)
                last_i = i + 1
        diag >>= self[last_i:]
        self = diag

        c_nodes = [tn.CopyNode(2, 2, f'c_input_{i}', dtype=complex)
                   for i in range(self.dom.count(bit))]
        q_nodes1 = [tn.CopyNode(2, 2, f'q1_input_{i}', dtype=complex)
                    for i in range(self.dom.count(qubit))]
        q_nodes2 = [tn.CopyNode(2, 2, f'q2_input_{i}', dtype=complex)
                    for i in range(self.dom.count(qubit))]

        inputs = [n[0] for n in c_nodes + q_nodes1 + q_nodes2]
        c_scan = [n[1] for n in c_nodes]
        q_scan1 = [n[1] for n in q_nodes1]
        q_scan2 = [n[1] for n in q_nodes2]
        nodes = c_nodes + q_nodes1 + q_nodes2
        for left, box, _ in self.inside:
            c_offset = left.count(bit)
            q_offset = left.count(qubit)
            if box == Circuit.swap(bit, bit):
                off = left.count(bit)
                c_scan[off], c_scan[off + 1] = c_scan[off + 1], c_scan[off]
            elif box == SWAP:
                off = left.count(qubit)
                for scan in (q_scan1, q_scan2):
                    scan[off], scan[off + 1] = scan[off + 1], scan[off]
            elif isinstance(box, Discard):
                assert box.n_qubits == 1
                tn.connect(q_scan1[q_offset], q_scan2[q_offset])
                del q_scan1[q_offset]
                del q_scan2[q_offset]
            elif box.is_mixed or isinstance(box, ClassicalGate):
                if isinstance(box, (Copy, Match, Measure, Encode)):
                    assert len(box.dom) == 1 or len(box.cod) == 1
                    node = tn.CopyNode(3, 2, 'cq_' + str(box), dtype=complex)
                else:
                    # only unoptimised gate is MixedState()
                    array = box.eval(mixed=True).array
                    node = tn.Node(array + 0j, 'cq_' + str(box))
                c_dom = box.dom.count(bit)
                q_dom = box.dom.count(qubit)
                c_cod = box.cod.count(bit)
                q_cod = box.cod.count(qubit)
                for i in range(c_dom):
                    tn.connect(c_scan[c_offset + i], node[i])
                for i in range(q_dom):
                    tn.connect(q_scan1[q_offset + i], node[c_dom + i])
                    tn.connect(q_scan2[q_offset + i], node[c_dom + q_dom + i])
                cq_dom = c_dom + 2 * q_dom
                c_edges = node[cq_dom:cq_dom + c_cod]
                q_edges1 = node[cq_dom + c_cod:cq_dom + c_cod + q_cod]
                q_edges2 = node[cq_dom + c_cod + q_cod:]
                c_scan[c_offset:c_offset + c_dom] = c_edges
                q_scan1[q_offset:q_offset + q_dom] = q_edges1
                q_scan2[q_offset:q_offset + q_dom] = q_edges2
                nodes.append(node)
            else:
                q_offset = left.count(qubit)
                utensor = box.array
                node1 = tn.Node(utensor + 0j, 'q1_' + str(box))
                with backend() as np:
                    node2 = tn.Node(np.conj(utensor) + 0j, 'q2_' + str(box))

                for i in range(len(box.dom)):
                    tn.connect(q_scan1[q_offset + i], node1[i])
                    tn.connect(q_scan2[q_offset + i], node2[i])

                edges1 = node1[len(box.dom):]
                edges2 = node2[len(box.dom):]
                q_scan1[q_offset:q_offset + len(box.dom)] = edges1
                q_scan2[q_offset:q_offset + len(box.dom)] = edges2
                nodes.extend([node1, node2])
        outputs = c_scan + q_scan1 + q_scan2
        return nodes, inputs + outputs

    def to_tk(self):
        """
        Export to t|ket>.

        Returns
        -------
        tk_circuit : pytket.Circuit
            A :class:`pytket.Circuit`.

        Note
        ----
        * No measurements are performed.
        * SWAP gates are treated as logical swaps.
        * If the circuit contains scalars or a :class:`Bra`,
          then :code:`tk_circuit` will hold attributes
          :code:`post_selection` and :code:`scalar`.

        Examples
        --------
        >>> from discopy.quantum import *

        >>> bell_test = H @ qubit >> CX >> Measure() @ Measure()
        >>> bell_test.to_tk()
        tk.Circuit(2, 2).H(0).CX(0, 1).Measure(0, 0).Measure(1, 1)

        >>> circuit0 = sqrt(2) @ H @ Rx(0.5) >> CX >> Measure() @ Discard()
        >>> circuit0.to_tk()
        tk.Circuit(2, 1).H(0).Rx(1.0, 1).CX(0, 1).Measure(0, 0).scale(2)

        >>> circuit1 = Ket(1, 0) >> CX >> qubit @ Ket(0) @ qubit
        >>> circuit1.to_tk()
        tk.Circuit(3).X(0).CX(0, 2)

        >>> circuit2 = X @ qubit ** 2\\
        ...     >> qubit @ SWAP >> CX @ qubit >> qubit @ SWAP
        >>> circuit2.to_tk()
        tk.Circuit(3).X(0).CX(0, 2)

        >>> circuit3 = Ket(0, 0)\\
        ...     >> H @ qubit\\
        ...     >> qubit @ X\\
        ...     >> CX\\
        ...     >> qubit @ Bra(0)
        >>> print(repr(circuit3.to_tk()))
        tk.Circuit(2, 1).H(0).X(1).CX(0, 1).Measure(1, 0).post_select({0: 0})
        """
        from discopy.quantum.tk import to_tk
        return to_tk(self)

    def to_pennylane(self, probabilities=False, backend_config=None,
                     diff_method='best'):
        """
        Export DisCoPy circuit to PennylaneCircuit.

        Parameters
        ----------
        probabilties : bool, default: False
            If True, the PennylaneCircuit will return the normalized
            probabilties of measuring the computational basis states
            when run. If False, it returns the unnormalized quantum
            states in the computational basis.
        backend_config : dict, default: None
            A dictionary of PennyLane backend configration options,
            including the provider (e.g. IBM or Honeywell), the device,
            the number of shots, etc. See the `PennyLane plugin
            documentation <https://pennylane.ai/plugins/>`_
            for more details.
        diff_method : str, default: "best"
            The differentiation method to use to obtain gradients for the
            PennyLane circuit. Some gradient methods are only compatible
            with simulated circuits. See the `PennyLane documentation
            <https://docs.pennylane.ai/en/stable/introduction/interfaces.html>`_
            for more details.

        Returns
        -------
        :class:`discopy.quantum.pennylane.PennylaneCircuit`
        """
        from discopy.quantum.pennylane import to_pennylane
        return to_pennylane(self, probabilities=probabilities,
                            backend_config=backend_config,
                            diff_method=diff_method)

    @staticmethod
    def from_tk(*tk_circuits):
        """
        Translate a :class:`pytket.Circuit` into a :class:`Circuit`, or
        a list of :class:`pytket` circuits into a :class:`Sum`.

        Parameters
        ----------
        tk_circuits : pytket.Circuit
            potentially with :code:`scalar` and
            :code:`post_selection` attributes.

        Returns
        -------
        circuit : :class:`Circuit`
            Such that :code:`Circuit.from_tk(circuit.to_tk()) == circuit`.

        Note
        ----
        * :meth:`Circuit.init_and_discard` is applied beforehand.
        * SWAP gates are introduced when applying gates to non-adjacent qubits.

        Examples
        --------
        >>> from discopy.quantum import *
        >>> import pytket as tk

        >>> c = Rz(0.5) @ qubit >> qubit @ Rx(0.25) >> CX
        >>> assert Circuit.from_tk(c.to_tk()) == c.init_and_discard()

        >>> tk_GHZ = tk.Circuit(3).H(1).CX(1, 2).CX(1, 0)
        >>> pprint = lambda c: print(str(c).replace(' >>', '\\n  >>'))
        >>> pprint(Circuit.from_tk(tk_GHZ))
        Ket(0)
          >> qubit @ Ket(0)
          >> qubit @ qubit @ Ket(0)
          >> qubit @ H @ qubit
          >> qubit @ CX
          >> SWAP @ qubit
          >> CX @ qubit
          >> SWAP @ qubit
          >> Discard(qubit) @ qubit @ qubit
          >> Discard(qubit) @ qubit
          >> Discard(qubit)
        >>> circuit = Ket(1, 0) >> CX >> qubit @ Ket(0) @ qubit
        >>> print(Circuit.from_tk(circuit.to_tk())[3:-3])
        X @ qubit @ qubit >> qubit @ SWAP >> CX @ qubit >> qubit @ SWAP

        >>> bell_state = Circuit.caps(qubit, qubit)
        >>> bell_effect = bell_state[::-1]
        >>> circuit = bell_state @ qubit >> qubit @ bell_effect >> Bra(0)
        >>> pprint(Circuit.from_tk(circuit.to_tk())[3:])
        H @ qubit @ qubit
          >> CX @ qubit
          >> qubit @ CX
          >> qubit @ H @ qubit
          >> Bra(0) @ qubit @ qubit
          >> Bra(0) @ qubit
          >> Bra(0)
          >> scalar(4)
        """
        # pylint: disable=import-outside-toplevel
        from discopy.quantum.tk import from_tk
        if not tk_circuits:
            return Sum([], qubit ** 0, qubit ** 0)
        if len(tk_circuits) == 1:
            return from_tk(tk_circuits[0])
        return sum(Circuit.from_tk(c) for c in tk_circuits)

    def grad(self, var, **params):
        """
        Gradient with respect to :code:`var`.

        Parameters
        ----------
        var : sympy.Symbol
            Differentiated variable.

        Returns
        -------
        circuit : `discopy.quantum.circuit.Sum`

        Examples
        --------
        >>> from math import pi
        >>> from sympy.abc import phi
        >>> from discopy.quantum import *
        >>> circuit = Rz(phi / 2) @ Rz(phi + 1) >> CX
        >>> assert circuit.grad(phi, mixed=False)\\
        ...     == (scalar(pi/2) @ Rz(phi/2 + .5) @ Rz(phi + 1) >> CX)\\
        ...     + (Rz(phi / 2) @ scalar(pi) @ Rz(phi + 1.5) >> CX)
        """
        return super().grad(var, **params)

    def jacobian(self, variables, **params):
        """
        Jacobian with respect to :code:`variables`.

        Parameters
        ----------
        variables : List[sympy.Symbol]
            Differentiated variables.

        Returns
        -------
        circuit : `discopy.quantum.circuit.Sum`
            with :code:`circuit.dom == self.dom`
            and :code:`circuit.cod == Digit(len(variables)) @ self.cod`.

        Examples
        --------
        >>> from sympy.abc import x, y
        >>> from discopy.quantum.gates import Bits, Ket, Rx, Rz
        >>> circuit = Ket(0) >> Rx(x) >> Rz(y)
        >>> assert circuit.jacobian([x, y])\\
        ...     == (Bits(0) @ circuit.grad(x)) + (Bits(1) @ circuit.grad(y))
        >>> assert not circuit.jacobian([])
        >>> assert circuit.jacobian([x]) == circuit.grad(x)
        """
        if not variables:
            return Sum([], self.dom, self.cod)
        if len(variables) == 1:
            return self.grad(variables[0], **params)
        from discopy.quantum.gates import Digits
        return sum(Digits(i, dim=len(variables)) @ self.grad(x, **params)
                   for i, x in enumerate(variables))

    def draw(self, **params):
        """ We draw the labels of a circuit whenever it's mixed. """
        draw_type_labels = params.get('draw_type_labels') or self.is_mixed
        params = dict({'draw_type_labels': draw_type_labels}, **params)
        return super().draw(**params)

    @staticmethod
    def permutation(perm, dom=None):
        dom = qubit ** len(perm) if dom is None else dom
        return frobenius.Diagram.permutation.__func__(Circuit, perm, dom)

    @staticmethod
    def cup_factory(left, right):
        from discopy.quantum.gates import CX, H, sqrt, Bra, Match, Discard

        if left == right == qubit:
            return CX >> H @ sqrt(2) @ qubit >> Bra(0, 0)
        if left == right == bit:
            return Match() >> Discard(bit)
        raise ValueError

    @staticmethod
    def spider_factory(n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None:
            raise NotImplementedError

        def factory(n_legs_in, n_legs_out, typ):
            if typ != qubit:
                raise NotImplementedError
            if (n_legs_in, n_legs_out) not in [(0, 1), (2, 1)]:
                return factory(n_legs_out, n_legs_in, qubit).dagger()
            from discopy.quantum.gates import CX, H, Bra, Ket, sqrt
            if (n_legs_in, n_legs_out) == (0, 1):
                return sqrt(2) >> Ket(0) >> H
            return CX >> qubit @ Bra(0)

        return frobenius.coherence(Circuit, factory)(
            n_legs_in, n_legs_out, typ)

    def apply_controlled(self, gate: Circuit, *indices: int) -> Circuit:
        """
        Post-compose with a controlled ``gate`` at given ``indices``.

        Parameters:
            gates : The gate to control.
            indices : The indices on which to apply the gate.
        """
        from discopy.quantum.gates import Controlled
        if min(indices) < 0 or max(indices) >= len(self.cod):
            raise IndexError
        if len(set(indices)) != len(indices):
            raise ValueError
        head, offset = indices[-1], min(indices)
        for x in sorted(filter(lambda x: x < head, indices), reverse=True):
            gate, head = Controlled(gate, distance=head - x), x
        head = indices[-1]
        for x in sorted(filter(lambda x: x > head, indices)):
            gate, head = Controlled(gate, distance=head - x), x
        return self\
            >> self.cod[:offset] @ gate @ self.cod[offset + len(gate.dom):]


class Box(tensor.Box[complex], Circuit):
    """
    A circuit box is a tensor box in a circuit diagram.

    Parameters:
        name : The name of the box.
        dom : The domain of the box.
        cod : The codomain of the box.
        data : The array inside the box.
        is_mixed : Whether the box is mixed.
    """

    def __init__(self, name: str, dom: Ty, cod: Ty,
                 data=None, is_mixed=True, **params):
        if not is_mixed:
            if all(isinstance(x, Digit) for x in (dom @ cod).inside):
                self.is_classical = True
            elif all(isinstance(x, Qudit) for x in (dom @ cod).inside):
                self.is_classical = False
            else:
                raise ValueError(messages.BOX_IS_MIXED)
        self._is_mixed = is_mixed
        tensor.Box[complex].__init__(self, name, dom, cod, data, **params)

    def __setstate__(self, state):
        if "_is_mixed" not in state:
            state["_is_mixed"] = state["_mixed"]
            del state["_mixed"]
        super().__setstate__(state)

    @property
    def array(self):
        """ The array of a quantum box. """
        if self.data is not None:
            with backend() as np:
                return np.array(self.data, dtype=complex).reshape(tuple(
                    obj.dim for obj in self.dom.inside + self.cod.inside))

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        raise NotImplementedError

    @property
    def is_mixed(self):
        return self._is_mixed

    def dagger(self):
        return self if self.is_dagger is None else super().dagger()

    def rotate(self, left=False):
        return self if self.z is None else super().rotate(left)


class Sum(tensor.Sum[complex], Box):
    """ Sums of circuits. """
    @property
    def is_mixed(self):
        return any(circuit.is_mixed for circuit in self.terms)

    def get_counts(self, backend=None, **params):
        if not self.terms:
            return {}
        if len(self.terms) == 1:
            return self.terms[0].get_counts(backend=backend, **params)
        counts = Circuit.get_counts(*self.terms, backend=backend, **params)
        result = {}
        for circuit_counts in counts:
            for bitstring, count in circuit_counts.items():
                result[bitstring] = result.get(bitstring, 0) + count
        return result

    def eval(self, backend=None, mixed=False, **params):
        mixed = mixed or any(t.is_mixed for t in self.terms)
        if not self.terms:
            return 0
        if len(self.terms) == 1:
            return self.terms[0].eval(backend=backend, mixed=mixed, **params)
        return sum(
            Circuit.eval(*self.terms, backend=backend, mixed=mixed, **params))

    def grad(self, var, **params):
        return sum(circuit.grad(var, **params) for circuit in self.terms)

    def to_tk(self):
        return [circuit.to_tk() for circuit in self.terms]


class Swap(tensor.Swap, Box):
    """ Implements swaps of circuit wires. """
    @property
    def is_mixed(self):
        return not isinstance(self.left.inside[0], type(self.right.inside[0]))

    @property
    def is_classical(self):
        return not self.is_mixed and isinstance(self.left.inside[0], Digit)

    def __str__(self):
        return "SWAP" if self.dom == qubit ** 2 else super().__str__()

    @property
    def array(self):
        left, = self.left.inside
        right, = self.right.inside
        return Tensor[complex].swap(Dim(left.dim), Dim(right.dim)).array


class Functor(frobenius.Functor):
    """ :class:`Circuit`-valued functor. """
    dom = cod = Category(Ty, Circuit)

    def __init__(self, ob, ar, dom=None, cod=None):
        if isinstance(ob, Mapping):
            ob = {x: qubit ** y if isinstance(y, int) else y
                  for x, y in ob.items()}
        super().__init__(ob, ar, dom=dom, cod=cod)


def index2bitstring(i: int, length: int) -> tuple[int, ...]:
    """ Turns an index into a bitstring of a given length. """
    if i >= 2 ** length:
        raise ValueError("Index should be less than 2 ** length.")
    if not i and not length:
        return ()
    return tuple(i >> k & 1 for k in range(length - 1, -1, -1))


def bitstring2index(bitstring):
    """ Turns a bitstring into an index. """
    return sum(value * 2 ** i for i, value in enumerate(bitstring[::-1]))


Circuit.braid_factory, Circuit.sum_factory = Swap, Sum
bit, qubit = Ty(Digit(2)), Ty(Qudit(2))
Id = Circuit.id
