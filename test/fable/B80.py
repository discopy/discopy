"""B80: Matrix.tensor/swap/id under jax, Function.id on a tuple wire, Functor(dtype=None).__repr__ and Channel.conjugate crash or leak (discopy/matrix.py:239, python/multiplicative.py:86, tensor.py:413, tensor.py:294).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy import frobenius
from discopy.matrix import Matrix, backend
from discopy.python import Function
from discopy.quantum.channel import Channel, Q
from discopy.tensor import Dim, Functor


def test_b80_matrix_tensor_under_jax():
    pytest.importorskip("jax")
    with backend('jax'):
        result = Matrix([1], 1, 1) @ Matrix([1], 1, 1)
    assert result.array.tolist() == [[1, 0], [0, 1]]


def test_b80_matrix_swap_under_jax():
    pytest.importorskip("jax")
    with backend('jax'):
        result = Matrix.swap(1, 1)
    assert result.array.tolist() == [[0, 1], [1, 0]]


def test_b80_matrix_id_under_jax_is_a_jax_array():
    jax = pytest.importorskip("jax")
    with backend('jax'):
        assert isinstance(Matrix.id(2).array, jax.Array)


def test_b80_function_id_on_a_tuple_wire():
    assert Function.id((tuple, ))((1, 2)) == (1, 2)


def test_b80_functor_with_no_dtype_has_a_repr():
    F = Functor({frobenius.Ty('x'): 2}, {}, dtype=None)
    assert isinstance(repr(F), str)


def test_b80_channel_conjugate_builds_a_channel():
    conjugate = Channel.id(Q(Dim(2))).conjugate(diagrammatic=False)
    assert isinstance(conjugate, Channel)
