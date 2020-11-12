# -*- coding: utf-8 -*-

""" DisCoPy quantum submodule: cqmap, circuit, gates, tk and zx. """

from discopy.quantum import cqmap, circuit, gates, zx
from discopy.quantum.cqmap import C, Q, CQ, CQMap
from discopy.quantum.circuit import (
    bit, qubit, Circuit, Id, Box, Sum, Swap, CircuitFunctor,
    Discard, MixedState, Measure, Encode)
from discopy.quantum.gates import (
    SWAP, ClassicalGate, QuantumGate, Ket, Bra, Bits,
    Rx, Rz, CU1, CRz, CRx, CZ, CX, X, Y, Z, H, S, T, scalar, sqrt,
    IQPansatz, random_tiling)
