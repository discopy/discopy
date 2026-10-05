import numpy as np
import pytest
from pytest import raises

from discopy.abc import Nat
from discopy.matrix import Matrix, backend
from discopy.utils import AxiomError


def test_Matrix_trace_left():
    f = Matrix[bool]([[1, 1], [0, 0]], 2, 2)
    assert f.trace() == Matrix[bool]([[1]], 1, 1)
    assert f.trace(left=True) == Matrix[bool]([[0]], 1, 1)
    assert Matrix[bool].swap(1, 1).trace(left=True) == Matrix[bool].id(1)


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
    pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float64
    assert Matrix([0.5j], dom=1, cod=1).dtype == np.complex128
    with backend('jax'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float32
    with backend('pytorch'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == torch.float32


def test_Matrix_copy():
    assert Matrix.copy(3, 2) == Matrix(
        [[1, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 1]], 3, 6)
    for x in range(4):
        for n in range(4):
            copy = Matrix.copy(x, n)
            assert (copy.dom, copy.cod) == (Nat(x), Nat(n * x))
            assert (copy.array == np.array(
                [[j % x == i for j in range(n * x)] for i in range(x)]
            ).reshape(x, n * x)).all()


def test_Matrix_comonoid():
    for x in range(4):
        identity, copy = Matrix.id(x), Matrix.copy(x, 2)
        assert copy >> identity @ Matrix.discard(x) == identity
        assert copy >> Matrix.discard(x) @ identity == identity
        assert copy >> Matrix.swap(x, x) == copy
        assert copy >> copy @ identity == Matrix.copy(x, 3)
        assert copy >> identity @ copy == Matrix.copy(x, 3)
        assert Matrix.ones(x) @ identity >> Matrix.merge(x, 2) == identity


def test_Matrix_copy_dtype():
    for x in range(4):
        dtype = Matrix.id(x).dtype
        assert Matrix.copy(x, 2).dtype == dtype
        assert Matrix.discard(x).dtype == Matrix.ones(x).dtype == dtype
    for dtype in (bool, int, float, complex):
        assert Matrix[dtype].copy(2, 2).dtype == dtype
