# -*- coding: utf-8 -*-
"""B15: the inherited eval is broken for zx diagrams (discopy/quantum/zx.py:37).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import numpy as np

from discopy.quantum.zx import Z, X


def test_b15_z_spider_is_identity():
    assert np.allclose(
        np.asarray(Z(1, 1).eval().array).reshape(2, 2), np.eye(2))


def test_b15_composition_evals():
    (Z(1, 1) >> X(1, 1)).eval()
