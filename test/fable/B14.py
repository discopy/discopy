# -*- coding: utf-8 -*-
"""B14: to_pyzx silently turns Y spiders into X spiders (discopy/quantum/zx.py:102).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import numpy as np

from discopy.quantum.zx import X, Y


def test_b14_y_spider_is_not_x():
    try:
        y_matrix = np.asarray(Y(1, 1, 0.25).to_pyzx().to_matrix())
    except NotImplementedError:
        return  # refusing to translate a Y spider is also correct
    x_matrix = np.asarray(X(1, 1, 0.25).to_pyzx().to_matrix())
    assert not np.allclose(y_matrix, x_matrix)
