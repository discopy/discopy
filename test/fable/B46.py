"""B46: >>, @ and + cast to the left operand's dtype, so Tensor.id(Dim(2)) >> v is zero for fractional v (discopy/tensor.py:143, discopy/matrix.py:246).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy.matrix import Matrix
from discopy.tensor import Dim, Tensor

v = Tensor([.5, .5, .5, .5], Dim(2), Dim(2))


def test_b46_id_on_the_left_is_the_identity():
    assert (Tensor.id(Dim(2)) >> v).array.tolist() == v.array.tolist()


def test_b46_swap_on_the_left_keeps_the_dtype():
    lhs = Tensor.swap(Dim(2), Dim(1)) >> v
    assert lhs.array.tolist() == v.array.tolist()


def test_b46_copy_on_the_left_keeps_the_dtype():
    lhs = Tensor.copy(Dim(2), 2) >> v @ v
    rhs = Tensor[float].copy(Dim(2), 2) >> v @ v
    assert lhs.array.tolist() == rhs.array.tolist()


def test_b46_id_tensor_on_the_left_keeps_the_dtype():
    assert (Tensor.id(Dim(1)) @ v).array.tolist() == v.array.tolist()


def test_b46_matrix_id_then_float_matrix():
    m = Matrix([.5, 1.5, 2.5, 3.5], 2, 2)
    assert (Matrix.id(2) >> m).array.tolist() == m.array.tolist()


def test_b46_matrix_add_promotes():
    lhs = Matrix([1, 1], 1, 2) + Matrix([.5, .5], 1, 2)
    assert lhs.array.tolist() == [[1.5, 1.5]]


def test_b46_complex_survives():
    w = Tensor([1j, 0, 0, 1j], Dim(2), Dim(2))
    assert (Tensor.id(Dim(2)) >> w).array.tolist() == w.array.tolist()
