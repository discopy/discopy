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
encoding each value of a time series. Each value induces a channel
:meth:`Reservoir.step` from memory to memory: prepare the input qubits
with :meth:`Reservoir.encode`, apply the unitary, then discard the input
qubits. The features of a time series are the Born probabilities of the
memory qubits after each step, computed by :meth:`Reservoir.run`; only the
linear readout :meth:`Reservoir.fit` is trained.
"""

from __future__ import annotations

from random import Random

from discopy.matrix import get_backend
from discopy.quantum.ansatze import Sim15ansatz
from discopy.quantum.circuit import Circuit, Id, qubit
from discopy.quantum.gates import Discard, Ket, Measure, Rx
from discopy.utils import assert_isinstance


class Reservoir:
    """
    A quantum reservoir is a fixed ``unitary`` circuit on ``memory`` qubits
    followed by ``inputs`` qubits driven by a time series.

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
        A reservoir with a :func:`discopy.quantum.ansatze.Sim15ansatz`
        unitary with phases drawn uniformly from a given ``seed``.

        Parameters:
            memory : The number of qubits kept between time steps.
            inputs : The number of qubits encoding each value of the series.
            depth : The number of layers of the ansatz.
            seed : The seed for the phases of the ansatz.
        """
        n, rng = memory + inputs, Random(seed)
        params = [rng.random() for _ in range(3)] if n == 1 else\
            [[rng.random() for _ in range(2 * n)] for _ in range(depth)]
        return cls(memory, inputs, Sim15ansatz(n, params))

    def encode(self, *values: float) -> Circuit:
        """
        Prepare the input qubits, one X rotation for each of ``values``
        given in half-turns.

        Parameters:
            values : As many numbers as there are input qubits.
        """
        if len(values) != self.inputs:
            raise ValueError(f"Expected {self.inputs} values, got {values}.")
        return Id().tensor(*(Ket(0) >> Rx(value) for value in values))

    def step(self, *values: float) -> Circuit:
        """
        The channel from memory to memory induced by input ``values``:
        encode them, apply the unitary, then discard the input qubits.

        Parameters:
            values : As many numbers as there are input qubits.
        """
        return Circuit.id(self.memory) @ self.encode(*values)\
            >> self.unitary\
            >> Circuit.id(self.memory) @ Discard(self.inputs)

    def run(self, sequence: list) -> list:
        """
        The Born probabilities of the memory qubits after each step,
        starting from the all-zero state.

        Parameters:
            sequence : A list of numbers, or tuples of numbers.
        """
        state = Ket(*self.memory * [0]).eval(mixed=True)
        readout = Measure(self.memory).eval()
        transition = (
            self.unitary >> Circuit.id(self.memory) @ Discard(self.inputs)
        ).eval()
        features = []
        for values in sequence:
            values = values if isinstance(values, tuple) else (values, )
            state = state @ self.encode(*values).eval(mixed=True)\
                >> transition
            features.append((state >> readout).array.real.reshape(-1))
        return features

    def fit(self, sequence: list, targets: list,
            regularisation: float = 1e-6):
        """
        The weights of the linear readout minimising the squared error
        between the features of ``sequence`` and ``targets``, with Tikhonov
        ``regularisation``.

        Parameters:
            sequence : A list of numbers, or tuples of numbers.
            targets : One number, or tuple of numbers, for each step.
            regularisation : The Tikhonov regularisation parameter.
        """
        if len(targets) != len(sequence):
            raise ValueError(
                f"Expected {len(sequence)} targets, got {len(targets)}.")
        if regularisation < 0:
            raise ValueError(
                f"Expected regularisation >= 0, got {regularisation}.")
        np = get_backend()
        features = np.stack(self.run(sequence))
        n_features = np.shape(features)[1]
        targets = np.reshape(
            np.array(targets, dtype=float), (len(features), -1))
        system = np.concatenate(
            (features, np.sqrt(regularisation) * np.eye(n_features)))
        values = np.concatenate(
            (targets, np.zeros((n_features, np.shape(targets)[1]))))
        return np.linalg.lstsq(system, values, rcond=None)[0]

    def predict(self, sequence: list, weights):
        """
        The linear readout ``weights`` applied to the features of
        ``sequence``.

        Parameters:
            sequence : A list of numbers, or tuples of numbers.
            weights : The matrix returned by :meth:`Reservoir.fit`.
        """
        np = get_backend()
        return np.stack(self.run(sequence)) @ weights
