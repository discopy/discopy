# -*- coding: utf-8 -*-
"""B22: eval crashes on Measure(override_bits=True) (discopy/quantum/channel.py:324).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.quantum.gates import Bits, Ket, Measure


def test_b22_measure_override_bits_evals():
    channel = (Ket(0) @ Bits(0) >> Measure(1, override_bits=True)).eval()
    assert channel.array is not None
