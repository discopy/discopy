# -*- coding: utf-8 -*-

import numpy as np
from pytest import raises

from discopy.quantum.ansatze import (
    IQPansatz, Sim14ansatz, Sim15ansatz)


def test_IQPAnsatz():
    with raises(ValueError):
        IQPansatz(10, np.array([]))


def test_Sim14Ansatz():
    with raises(ValueError):
        Sim14ansatz(10, np.array([]))


def test_Sim15Ansatz():
    with raises(ValueError):
        Sim15ansatz(10, np.array([]))
