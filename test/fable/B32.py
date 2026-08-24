"""B32: tensor.Box.setoid puts raw arrays in the comparison tuple (discopy/tensor.py:743).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import numpy as np

from discopy import tensor
from discopy.tensor import Dim


def test_b32_array_data_compares():
    a = tensor.Box('h', Dim(2), Dim(2), np.array([0, 1, 1, 0]))
    b = tensor.Box('h', Dim(2), Dim(2), np.array([0, 1, 1, 1]))
    assert (a == b) is False


def test_b32_nested_list_data_hashes():
    assert isinstance(
        hash(tensor.Box('f', Dim(2), Dim(2), [[0, 1], [1, 0]])), int)
