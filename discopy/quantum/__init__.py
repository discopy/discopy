# -*- coding: utf-8 -*-

""" DisCoPy quantum submodule: cqmap, circuit, gates, tk and zx. """

from discopy.quantum import cqmap, circuit, gates, optics, zx
from discopy.quantum.cqmap import C, Q, CQ, CQMap
from discopy.quantum.circuit import (
    bit, qubit, Digit, Qudit, Circuit, Id, Box, Sum, Swap,
    Functor as CircuitFunctor,
    Discard, MixedState, Measure, Encode, IQPansatz, Sim14ansatz, Sim15ansatz,
    Sim8ansatz, random_tiling, real_amp_ansatz)
from discopy.quantum.gates import (
    SWAP, ClassicalGate, QuantumGate, Controlled, Ket, Bra, Bits, Copy, Match,
    Rx, Ry, Rz, CU1, CRz, CRx, CZ, CX, X, Y, Z, H, S, T, scalar, sqrt, rewire)
