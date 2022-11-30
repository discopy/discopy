# -*- coding: utf-8 -*-

"""
quantum
=======

DisCoPy quantum modules: channel, circuit, gates, tk and zx.

.. autosummary::
    :template: module.rst
    :toctree: api

    discopy.quantum.channel
    discopy.quantum.circuit
    discopy.quantum.gates
    discopy.quantum.ansatze
    discopy.quantum.optics
    discopy.quantum.zx
    discopy.quantum.tk
    discopy.quantum.pennylane
"""

from discopy.quantum import circuit, gates, channel  #, ansatze, optics, zx
from discopy.quantum.channel import C, Q, CQ, Channel
from discopy.quantum.circuit import (
    bit, qubit, Digit, Qudit, Circuit, Id, Box, Sum, Swap,
    Functor as CircuitFunctor,
)
from discopy.quantum.gates import (
    Discard, MixedState, Measure, Encode,
    SWAP, ClassicalGate, QuantumGate,
    Controlled, Ket, Bra, Bits, Copy, Match,
    Rx, Ry, Rz, CU1, CRz, CRx, CZ, CX,
    X, Y, Z, H, S, T, scalar, sqrt, rewire,
)
# from discopy.quantum.ansatze import (
#     IQPansatz, Sim14ansatz, Sim15ansatz,
#     random_tiling, real_amp_ansatz,
# )
