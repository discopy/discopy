# -*- coding: utf-8 -*-

"""
Quantum reservoir computing, i.e. supervised learning on time series where
the features are the measurement statistics of a fixed quantum system.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Reservoir

Note
----
A quantum reservoir in the sense of Fujii and Nakajima
:cite:`FujiiNakajima17` is a discrete-time dynamical system in the category
of classical-quantum channels: a fixed ``unitary`` acts on ``memory``
qubits kept between time steps together with ``inputs`` fresh qubits
encoding each value of a time series. It is a monoidal stream
:class:`discopy.stream.Stream` of circuits, :meth:`Reservoir.stream`, from
the input qubits to nothing with the memory as feedback, and a time series
is the stream of state preparations :meth:`Reservoir.encoding` composed
with it. Its slice at each time step is a channel from memory to memory
:meth:`Reservoir.step`: prepare the input qubits with
:meth:`Reservoir.encode`, apply the unitary, then discard the input qubits.
The features of a time series are the Born probabilities of the memory
qubits after each step, computed by :meth:`Reservoir.run`; only the
linear readout :meth:`Reservoir.fit` is trained, with an optional bias.
"""

from __future__ import annotations

from random import Random

from discopy.matrix import get_backend
from discopy.quantum.ansatze import Sim15ansatz
from discopy.quantum.circuit import Circuit, Id, qubit
from discopy.quantum.gates import Discard, Ket, Measure, Rx
from discopy.stream import Stream, Ty
from discopy.utils import assert_isinstance


class Reservoir:
    """
    A quantum reservoir is a fixed unitary on memory and input qubits.

    Parameters:
        memory : The number of qubits kept between time steps.
        inputs : The number of qubits encoding each value of the series.
        unitary : A circuit on ``memory + inputs`` qubits.

    Example
    -------
    We train the readout of a random reservoir to recall the previous
    value of a sequence, beating the variance of the uniform distribution:

    >>> from random import Random
    >>> rng = Random(0)
    >>> sequence = [rng.random() for _ in range(30)]
    >>> targets = [0.] + sequence[:-1]
    >>> reservoir = Reservoir.random(memory=2, seed=42)
    >>> weights = reservoir.fit(sequence, targets)
    >>> predictions = reservoir.predict(sequence, weights)[:, 0]
    >>> errors = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    >>> assert sum(errors) / len(errors) < 1 / 12

    The unitary can be any circuit, e.g. an array of neutral atoms evolving
    under the Rydberg Hamiltonian of Pasqal's QPU between the steps, as
    proposed by Araiza Bravo, Najafi, Gao and Yelin :cite:`BravoEtAl22`:

    >>> from discopy.quantum.ansatze import Rydberg
    >>> rydberg = Reservoir(memory=2, inputs=1, unitary=Rydberg(
    ...     [(0, 0), (0, 8), (8, 0)], duration=0.5, omega=4, delta=2))
    >>> weights = rydberg.fit(sequence, targets)
    >>> predictions = rydberg.predict(sequence, weights)[:, 0]
    >>> errors = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    >>> assert sum(errors) / len(errors) < 1 / 12
    """
    def __init__(self, memory: int, inputs: int, unitary: Circuit):
        assert_isinstance(unitary, Circuit)
        if memory < 0 or inputs < 0:
            raise ValueError(
                f"Expected memory, inputs >= 0, got {memory}, {inputs}.")
        if unitary.is_mixed or unitary.dom != unitary.cod\
                or unitary.dom != qubit ** (memory + inputs):
            raise ValueError(
                f"Expected a unitary on {memory + inputs} qubits, "
                f"got {unitary.dom} to {unitary.cod}.")
        np, dim = get_backend(), 2 ** (memory + inputs)
        array = np.reshape(unitary.eval().array, (dim, dim))
        if not np.allclose(array.conj().T @ array, np.eye(dim)):
            raise ValueError(f"Expected a unitary, got {unitary}.")
        self.memory, self.inputs, self.unitary = memory, inputs, unitary

    def __eq__(self, other):
        return isinstance(other, Reservoir)\
            and (self.memory, self.inputs, self.unitary)\
            == (other.memory, other.inputs, other.unitary)

    def __repr__(self):
        return f"Reservoir(memory={self.memory}, "\
            f"inputs={self.inputs}, unitary={repr(self.unitary)})"

    @classmethod
    def random(cls, memory: int, inputs: int = 1,
               depth: int = 2, seed: int = 0) -> Reservoir:
        """
        A reservoir with a random Sim15ansatz unitary.

        Its phases are drawn uniformly from a given ``seed``.

        Parameters:
            memory : The number of qubits kept between time steps.
            inputs : The number of qubits encoding each value of the series.
            depth : The number of layers of the ansatz.
            seed : The seed for the phases of the ansatz.
        """
        n, rng = memory + inputs, Random(seed)
        if n == 1:
            return cls(memory, inputs, Id(1).then(*[
                Sim15ansatz(1, [rng.random() for _ in range(3)])
                for _ in range(depth)]))
        params = [[rng.random() for _ in range(2 * n)] for _ in range(depth)]
        return cls(memory, inputs, Sim15ansatz(n, params))

    def encode(self, *values: float) -> Circuit:
        """
        Prepare the input qubits, one X rotation per value in half-turns.

        Parameters:
            values : As many numbers as there are input qubits.
        """
        if len(values) != self.inputs:
            raise ValueError(f"Expected {self.inputs} values, got {values}.")
        return Id().tensor(*(Ket(0) >> Rx(value) for value in values))

    @property
    def cell(self) -> Circuit:
        """
        The channel from the input qubits and the memory to the memory.

        Apply the unitary, then discard the input qubits.
        """
        return Circuit.swap(qubit ** self.inputs, qubit ** self.memory)\
            >> self.unitary\
            >> Circuit.id(self.memory) @ Discard(self.inputs)

    def step(self, *values: float) -> Circuit:
        """
        The channel from memory to memory induced by input ``values``.

        Encode them, apply the unitary, then discard the input qubits.

        Parameters:
            values : As many numbers as there are input qubits.
        """
        return self.encode(*values) @ Circuit.id(self.memory) >> self.cell

    @property
    def stream(self) -> Stream[Circuit]:
        """
        The reservoir as a stream from the input qubits to nothing.

        The memory is fed back from each time step to the next, starting
        from the all-zero state, so that its slice at a later time step is
        the :meth:`Reservoir.cell`.

        Example
        -------
        >>> reservoir = Reservoir.random(memory=2)
        >>> stream = reservoir.stream
        >>> print(stream.now.dom, "->", stream.now.cod)
        qubit -> qubit @ qubit
        >>> assert stream.later.now == stream.later.later.now == reservoir.cell
        """
        inputs, memory = (
            Ty[Circuit.ob](qubit ** n) for n in (self.inputs, self.memory))
        first = Circuit.id(self.inputs) @ Ket(*self.memory * [0])\
            >> self.cell
        body = Stream[Circuit](
            first, inputs @ memory.delay(), memory,
            _later=lambda: Stream[Circuit](self.cell))
        return body.feedback(dom=inputs, cod=Ty[Circuit.ob](), mem=memory)

    def encoding(self, sequence: list) -> Stream[Circuit]:
        """
        The stream of state preparations of the input qubits for a sequence.

        It is defined on as many time steps as the sequence has values.

        Parameters:
            sequence : A list of numbers, or tuples of numbers.

        Example
        -------
        >>> reservoir = Reservoir.random(memory=2)
        >>> stream = reservoir.encoding([0.1, 0.2]) >> reservoir.stream
        >>> print(stream.now.dom, "->", stream.now.cod)
        Ty() -> qubit @ qubit
        """
        values, *rest = sequence
        values = values if isinstance(values, tuple) else (values, )
        inputs = Ty[Circuit.ob](qubit ** self.inputs)
        return Stream[Circuit](
            self.encode(*values), Ty[Circuit.ob](), inputs,
            _later=lambda: self.encoding(rest))

    def run(self, sequence: list, bias: bool = False) -> list:
        """
        The Born probabilities of the memory qubits after each step.

        Walks the slices of :meth:`Reservoir.encoding` and
        :meth:`Reservoir.stream` together, i.e. of their composite, so that
        the memory is read between time steps, evaluating each distinct
        slice of the reservoir once.

        Parameters:
            sequence : A list of numbers, or tuples of numbers.
            bias : Whether to end each list of features with a constant one.
        """
        if not sequence:
            return []
        np = get_backend()
        readout = Measure(self.memory).eval()
        inputs, reservoir = self.encoding(sequence), self.stream
        state, channels, features = Ket().eval(mixed=True), {}, []
        for i in range(len(sequence)):
            if reservoir.now not in channels:
                channels[reservoir.now] = reservoir.now.eval(mixed=True)
            state = inputs.now.eval(mixed=True) @ state\
                >> channels[reservoir.now]
            probabilities = (state >> readout).array.real.reshape(-1)
            features.append(np.concatenate((probabilities, np.ones(1)))
                            if bias else probabilities)
            if i + 1 < len(sequence):
                inputs, reservoir = inputs.later, reservoir.later
        return features

    def fit(self, sequence: list, targets: list,
            regularisation: float = 1e-6, bias: bool = True):
        """
        The weights of the linear readout, fit by ridge regression.

        Minimises the squared error between the features of ``sequence``
        and ``targets``, with Tikhonov ``regularisation``. With ``bias``,
        the last row of the weights is an intercept, which the
        regularisation leaves out.

        Parameters:
            sequence : A list of numbers, or tuples of numbers.
            targets : One number, or tuple of numbers, for each step.
            regularisation : The Tikhonov regularisation parameter.
            bias : Whether to fit an intercept as well.

        Example
        -------
        The bias absorbs a constant offset of the targets:

        >>> reservoir = Reservoir.random(memory=1, seed=1)
        >>> sequence = [0.1, 0.5, 0.9, 0.3]
        >>> weights = reservoir.fit(sequence, len(sequence) * [7.], bias=True)
        >>> predictions = reservoir.predict(sequence, weights, bias=True)
        >>> assert abs(predictions - 7).max() < 1e-3
        """
        if len(targets) != len(sequence):
            raise ValueError(
                f"Expected {len(sequence)} targets, got {len(targets)}.")
        if regularisation < 0:
            raise ValueError(
                f"Expected regularisation >= 0, got {regularisation}.")
        np = get_backend()
        features = np.stack(self.run(sequence, bias))
        n_features = np.shape(features)[1]
        targets = np.reshape(
            np.array(targets, dtype=float), (len(features), -1))
        penalty = np.diag(np.array(
            (n_features - 1) * [1.] + [0. if bias else 1.]))
        system = np.concatenate(
            (features, np.sqrt(regularisation) * penalty))
        values = np.concatenate(
            (targets, np.zeros((n_features, np.shape(targets)[1]))))
        return np.linalg.lstsq(system, values, rcond=None)[0]

    def predict(self, sequence: list, weights, bias: bool = True):
        """
        The linear readout applied to the features of ``sequence``.

        Parameters:
            sequence : A list of numbers, or tuples of numbers.
            weights : The matrix returned by :meth:`Reservoir.fit`.
            bias : Whether the weights end with an intercept.
        """
        np = get_backend()
        return np.stack(self.run(sequence, bias)) @ weights
