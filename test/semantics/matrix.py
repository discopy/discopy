import numpy as np
from pytest import raises

from discopy.matrix import Matrix, backend
from discopy.utils import AxiomError


def test_bad_composition():
    m = Matrix([1, 2, 3, 4, 5, 6], 2, 3)

    with raises(TypeError):
        m >> 1
    with raises(AxiomError):
        m >> m


def test_matrix_tensor():
    m = Matrix([1], 1, 1)
    assert (m.tensor(m, m).array == np.eye(3)).all()
    with raises(TypeError):
        m @ "bla"


def test_matrix_add():
    m = Matrix([1, 2, 3, 4, 5, 6], 2, 3)
    assert 0 + m == m
    with raises(TypeError):
        m + 123
    with raises(AxiomError):
        m + m.dagger()


def test_repeat():
    with raises(TypeError):
        Matrix[int](0, 1, 1, 0).repeat()


def test_autotyping():
    import torch
    import tensorflow as tf
    assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float64
    assert Matrix([0.5j], dom=1, cod=1).dtype == np.complex128
    with backend('jax'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float32
    with backend('pytorch'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == torch.float32
    with backend('tensorflow'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == tf.float64

