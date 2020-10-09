# -*- coding: utf-8 -*-

from pytest import raises
from unittest.mock import Mock
from discopy.quantum import *

def test_Circuit_to_tk():
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Id(1) @ bell_effect)[::-1]
    tk_circ = snake.to_tk()
    assert repr(tk_circ).split('.')[2:-2] == ['H(1)',
        'CX(1, 2)',
        'CX(0, 1)',
        'Measure(1, 1)',
        'H(0)',
        'Measure(0, 0)',
        'post_select({0: 0, 1: 0})']
    assert np.isclose(tk_circ.scalar, 2)

def test_Circuit_get_counts_snake():
    backend = Mock()
    backend.get_counts.return_value = {
        (0, 0, 0): 251, (0, 0, 1): 247,
        (1, 1, 0): 268, (1, 1, 1): 258}
    scaled_bell = Circuit.caps(qubit, qubit)
    snake = scaled_bell @ Id(1) >> Id(1) @ scaled_bell[::-1]
    result = np.round(snake.eval(backend, seed=42).array)
    expected = np.round((Ket(0) >> snake).measure())
    assert np.all(result == expected)

def test_Circuit_get_counts_empty():
    backend = Mock()
    backend.get_counts.return_value = {}
    with raises(RuntimeError):
        Id(1).get_counts(backend)
