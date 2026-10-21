# -*- coding: utf-8 -*-

import numpy as np
import pytest
from pytest import raises

from discopy import compact, quantum
from discopy.quantum import Discard, Id, Ket, Measure
from discopy.quantum.reservoir import Reservoir


def test_Reservoir_errors():
    with raises(ValueError):
        Reservoir(2, 1, Id(2))
    with raises(ValueError):
        Reservoir(1, 1, Ket(0, 0))
    with raises(ValueError):
        Reservoir(-1, 2, Id(1))
    with raises(ValueError):
        Reservoir(2, -1, Id(1))
    with raises(ValueError):
        Reservoir(1, 0, Discard(1) @ Ket(0))
    with raises(ValueError):
        Reservoir(1, 0, quantum.Bra(0) >> Ket(0))
    with raises(ValueError):
        Reservoir.random(1).encode(0.1, 0.2)
    with raises(ValueError):
        Reservoir.random(1).fit([0.1, 0.2], [0.1, 0.2, 0.3])
    with raises(ValueError):
        Reservoir.random(1).fit([0.1], [0.1], regularisation=-1)


def test_Reservoir_random_depth():
    assert Reservoir.random(memory=1, inputs=0, depth=0).unitary == Id(1)
    assert Reservoir.random(memory=2, inputs=1, depth=0).unitary == Id(3)
    with raises(ValueError):
        Reservoir.random(memory=1, inputs=0, depth=-1)
    assert len(Reservoir.random(memory=1, inputs=0, depth=3).unitary) == 9


def test_Reservoir_repr():
    reservoir = Reservoir.random(memory=1, inputs=1, seed=5)
    scope = {"Reservoir": Reservoir,
             "quantum": quantum, "compact": compact,
             **vars(quantum.gates)}
    assert eval(repr(reservoir), scope) == reservoir
    assert reservoir == Reservoir.random(memory=1, inputs=1, seed=5)
    assert reservoir != Reservoir.random(memory=1, inputs=1, seed=6)


def test_Reservoir_step():
    reservoir = Reservoir.random(memory=1, inputs=1, seed=3)
    state = Ket(0).eval(mixed=True) >> reservoir.step(0.3).eval()
    assert np.allclose(
        reservoir.run([0.3])[0],
        (state >> Measure().eval()).array.real.reshape(-1))


def test_Reservoir_run():
    reservoir = Reservoir.random(memory=2, inputs=2, depth=1, seed=1)
    features = reservoir.run(2 * [(0.1, 0.2)])
    assert np.shape(features) == (2, 4)
    for feature in features:
        assert all(feature >= 0) and abs(sum(feature) - 1) < 1e-12


def test_Reservoir_fit():
    reservoir = Reservoir.random(memory=2, seed=42)
    sequence = [i / 7 % 1 for i in range(20)]
    targets = [0.] + sequence[:-1]
    readout = reservoir.fit(sequence, targets)
    predictions = reservoir.predict(sequence, readout)[:, 0]
    baseline = sum((t - np.mean(targets)) ** 2 for t in targets)
    assert sum((predictions - targets) ** 2) < baseline


def test_Reservoir_stream_unrolls_to_run():
    reservoir = Reservoir.random(memory=2, inputs=1, depth=1, seed=2)
    sequence = [0.2, 0.9, 0.4]
    stream = reservoir.encoding(sequence) >> reservoir.stream
    for step, feature in enumerate(reservoir.run(sequence)):
        state = stream.unroll(step).now.eval(mixed=True)
        assert np.allclose(
            feature, (state >> Measure(2).eval()).array.real.reshape(-1))


def test_Reservoir_bias():
    reservoir = Reservoir.random(memory=1, seed=4)
    sequence, targets = [0.3, 0.6, 0.1], [1., 2., 3.]
    readout = reservoir.fit(sequence, targets, bias=False)
    assert np.shape(readout.weights) == (2, 1)
    assert np.allclose(readout.intercept, 0)
    offset = [t + 100 for t in targets]
    shifted = reservoir.fit(sequence, offset, regularisation=1.)
    unshifted = reservoir.fit(sequence, targets, regularisation=1.)
    assert np.allclose(shifted.weights, unshifted.weights)
    assert np.allclose(shifted.intercept, unshifted.intercept + 100)
    assert np.allclose(reservoir.predict(sequence, shifted),
                       reservoir.predict(sequence, unshifted) + 100)


def test_Reservoir_encode():
    reservoir = Reservoir.random(memory=1)
    probabilities = [
        (reservoir.encode(x) >> Measure()).eval().array.real[1]
        for x in np.linspace(0, 1, 11)]
    assert np.all(np.diff(probabilities) > 0)


def test_Reservoir_empty_sequence():
    assert Reservoir.random(memory=1).run([]) == []


def test_Reservoir_fit_on_jax():
    from discopy.matrix import backend
    pytest.importorskip("jax")
    reservoir = Reservoir.random(memory=1, seed=4)
    with backend("jax"):
        readout = reservoir.fit([0.3, 0.6, 0.1], [1., 2., 3.])
    assert np.shape(readout.weights) == (2, 1)
