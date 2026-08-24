"""B5: Matrix.copy is wrong for x, n >= 2 (discopy/matrix.py:325).
Asserts the correct behaviour, red while the bug is live — issue #606."""

from discopy.matrix import Matrix


def test_b5_copy_array():
    assert (Matrix.copy(2, 2).array == [[1, 0, 1, 0], [0, 1, 0, 1]]).all()


def test_b5_copy_counit():
    lhs = Matrix.copy(2, 2) >> Matrix.discard(2) @ Matrix.id(2)
    assert (lhs.array == Matrix.id(2).array).all()


def test_b5_copy_cocommutative():
    lhs = Matrix.copy(2, 2) >> Matrix.swap(2, 2)
    assert (lhs.array == Matrix.copy(2, 2).array).all()
