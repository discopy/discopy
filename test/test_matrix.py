from pytest import raises

from discopy.matrix import Matrix, block_diag
from discopy import PRO, Sum, Dim
from discopy.cat import AxiomError

import numpy as np


def test_bad_composition():
    m = Matrix(PRO(2), PRO(3), [1, 2, 3, 4, 5, 6])

    with raises(TypeError):
        m >> 1
    with raises(AxiomError):
        m >> m


def test_matrix_tensor():
    m = Matrix(PRO(1), PRO(1), [1])
    assert (m.tensor(m, m).array == np.eye(3)).all()
    with raises(TypeError):
        m @ 1


def test_matrix_add():
    m = Matrix(PRO(2), PRO(3), [1, 2, 3, 4, 5, 6])
    assert m + 0 == 0 + m == m
    with raises(TypeError):
        m + 123
    with raises(AxiomError):
        m + m.dagger()


def test_bad_swap():
    with raises(NotImplementedError):
        Matrix.swap(PRO(1), PRO(2))


def test_block_diag():
    assert np.all(block_diag() == np.array([]))
    with raises(ValueError):
        block_diag([[[1]]])


def test_Tensor_then():
    assert Matrix(PRO(1), PRO(1), [1])
    m = Matrix(PRO(2), PRO(2), [0, 1, 1, 0])
    assert repr(m) == str(m)\
        == "Matrix(dom=PRO(2), cod=PRO(2), array=[0, 1, 1, 0])"
    u = Matrix(PRO(2), PRO(2), [1, 0, 0, 0])
    v = Matrix(PRO(2), PRO(2), [0, 0, 0, 1])
    assert u + v == Matrix.id(PRO(2))
    assert Sum([m >> v, m >> u]) == m >> Sum([v, u])
