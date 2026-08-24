"""B9: Matrix.map casts to the old dtype and composition drops it (discopy/matrix.py:315).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.matrix import Matrix
from discopy.tensor import Box, Dim


def test_b9_map_dtype():
    mapped = Matrix([1, 2], 1, 2).map(lambda x: x + 0.5)
    assert mapped.array.tolist() == [[1.5, 2.5]]


def test_b9_composition_keeps_dtype():
    f = Box('f', Dim(2), Dim(2), [1, 0, 0, 1])
    assert (f >> f).eval().dtype == f.eval().dtype
