# -*- coding: utf-8 -*-

"""
DisCoPy quantum modules: channel, circuit, gates, tk and zx.
"""

from discopy.quantum import circuit, gates, channel, ansatze, zx
from discopy.quantum.ansatze import (
    IQPansatz, Sim14ansatz, Sim15ansatz,
)
from discopy.quantum.channel import C, Q, CQ, Channel
from discopy.quantum.circuit import (
    bit, qubit, Ty, Digit, Qudit, Circuit, Id, Box, Sum, Swap,
    Functor as CircuitFunctor,
)
from discopy.quantum.gates import (
    Discard, MixedState, Measure, Encode,
    SWAP, ClassicalGate, QuantumGate,
    Controlled, Ket, Bra, Bits, Copy, Match,
    Rx, Ry, Rz, CU1, CRz, CRx, CZ, CX,
    X, Y, Z, H, S, T, scalar, sqrt,
)
