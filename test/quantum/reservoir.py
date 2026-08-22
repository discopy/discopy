# -*- coding: utf-8 -*-

import numpy as np
from pytest import raises

from discopy import compact, quantum
from discopy.quantum import Id, Ket, Measure
from discopy.quantum.reservoir import Reservoir


def test_Reservoir_errors():
    with raises(ValueError):
        Reservoir(2, 1, Id(2))
    with raises(ValueError):
        Reservoir(1, 1, Ket(0, 0))
    with raises(ValueError):
        Reservoir.random(1).encode(0.1, 0.2)
    with raises(ValueError):
        Reservoir.random(1).fit([0.1, 0.2], [0.1, 0.2, 0.3])
    with raises(ValueError):
        Reservoir.random(1).fit([0.1], [0.1], regularisation=-1)


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
    weights = reservoir.fit(sequence, targets)
    predictions = reservoir.predict(sequence, weights)[:, 0]
    baseline = sum((t - np.mean(targets)) ** 2 for t in targets)
    assert sum((predictions - targets) ** 2) < baseline
