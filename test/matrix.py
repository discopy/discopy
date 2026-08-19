from itertools import permutations

import numpy as np
import pytest
from pytest import raises

from discopy.matrix import Matrix, backend
from discopy.symmetric import Ty, Permutation, Functor
from discopy.tensor import Tensor, Dim
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
    pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float64
    assert Matrix([0.5j], dom=1, cod=1).dtype == np.complex128
    with backend('jax'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == np.float32
    with backend('pytorch'):
        assert Matrix([0.5, 0.5], dom=1, cod=2).dtype == torch.float32



def test_permutation():
    """ The native permutation agrees with a composition of swaps. """
    def swaps(xs, doms):
        result, done, cur = Matrix.id(sum(doms)), 0, list(doms)
        xs = list(xs)
        while xs != list(range(len(xs))):
            i = xs[0]
            left = sum(cur[:i])
            result >>= Matrix.id(done) @ Matrix.swap(left, cur[i])\
                @ Matrix.id(sum(cur[i + 1:]))
            done, cur = done + cur[i], cur[:i] + cur[i + 1:]
            xs = [x - 1 if x > i else x for x in xs[1:]]
        return result

    assert Matrix.permutation([1, 0], [1, 1]) == Matrix.swap(1, 1)
    for xs in permutations(range(4)):
        doms = [1 + x % 3 for x in xs]
        assert Matrix.permutation(xs, doms) == swaps(xs, doms)
    with raises(ValueError):
        Matrix.permutation([0, 0], [1, 1])


def test_copy_defaults_to_two():
    """ Matrix is a MarkovCategory, whose copy has n=2 by default. """
    assert Matrix.copy(2) == Matrix.copy(2, 2)
    assert Tensor.copy(Dim(2)) == Tensor.copy(Dim(2), 2)


def test_permutation_doms_optional():
    """ One dimension per block is the default and takes the short path. """
    for size in range(1, 6):
        for xs in permutations(range(size)):
            assert Matrix.permutation(xs) == Matrix.permutation(xs, size * [1])
    assert Matrix.permutation([1, 0]) == Matrix.swap(1, 1)
    assert Matrix.permutation([0, 1]) == Matrix.id(2)
    with raises(ValueError):
        Matrix.permutation([0, 2])


def test_matrix_valued_functor_on_permutation():
    """ Matrix is a MarkovCategory, so it is a valid symmetric codomain. """
    x, y = Ty('x'), Ty('y')
    perm = Permutation(x @ y @ x, [1, 2, 0])
    F = Functor(ob_map={x: 2, y: 3}, ar_map={}, cod=Matrix)
    assert F(perm) == Matrix.permutation([1, 2, 0], [2, 3, 2])
