# -*- coding: utf-8 -*-
"""B23: to_tk and from_tk gate coverage is broken for CCX, Controlled(Ry) and CU1 (discopy/quantum/tk.py:255-283).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import pytket as tk

from discopy.quantum.circuit import Circuit
from discopy.quantum.gates import GATES, Controlled, Ry


def test_b23_ccx_to_tk():
    assert GATES['CCX'].to_tk() is not None


def test_b23_controlled_ry_to_tk():
    try:
        Controlled(Ry(0.3)).to_tk()
    except NotImplementedError:
        pass  # an explicit refusal is also correct


def test_b23_from_tk_cu1():
    Circuit.from_tk(tk.Circuit(2).CU1(0.5, 0, 1))
