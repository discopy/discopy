"""B36: Matrix repr mutates numpy's global printoptions and elides entries past 16 (discopy/matrix.py:409).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import numpy

from discopy.matrix import Matrix


def test_b36_repr_leaves_printoptions_alone():
    numpy.set_printoptions(threshold=1000)
    try:
        repr(Matrix(list(range(25)), 5, 5))
        assert numpy.get_printoptions()['threshold'] == 1000
    finally:
        numpy.set_printoptions(threshold=1000)


def test_b36_repr_roundtrips():
    matrix = Matrix(list(range(25)), 5, 5)
    namespace = {'Matrix': Matrix, 'int64': numpy.int64}
    assert eval(repr(matrix), namespace) == matrix
