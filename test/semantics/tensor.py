import numpy as np
import tensornetwork as tn
from pytest import raises

from discopy.utils import AxiomError
from discopy.tensor import *
from discopy import frobenius


def test_backend():
    import jax.numpy
    import torch
    import tensorflow.experimental.numpy as tnp
    assert isinstance(Tensor.id().array, np.ndarray)
    with backend('jax'):
        assert isinstance(Tensor.id().array, jax.numpy.ndarray)
        with backend('pytorch'):
            assert isinstance(Tensor.id().array, torch.Tensor)
            with backend('tensorflow'):
                assert isinstance(Tensor.id().array, tnp.ndarray)
            assert isinstance(Tensor.id().array, torch.Tensor)
        assert isinstance(Tensor.id().array, jax.numpy.ndarray)
    assert isinstance(Tensor.id().array, np.ndarray)


def test_Tensor_repr_with_tf():
    with backend('tensorflow'):
        alice = Tensor([1, 2], Dim(1), Dim(2))
        assert repr(alice)\
            == "Tensor[<dtype: 'int64'>]([1, 2], dom=Dim(1), cod=Dim(2))"


def test_Dim():
    with raises(TypeError):
        Dim('a')
    with raises(ValueError):
        Dim(-1)
    dim = Dim(2, 3)
    assert Dim(1) @ dim == dim @ Dim(1) == dim
    assert Dim(1).tensor(*(Dim(2, 3), Dim(4), Dim(1))) == Dim(2, 3, 4)
    assert dim[:1] == Dim(3, 2)[1:] == Dim(2)
    assert dim[0] == Dim(3, 2)[1]
    assert dim.inside[0] == 2
    assert repr(Dim(1, 2, 3)) == str(dim) == "Dim(2, 3)"
    assert {dim: 42}[dim] == 42
    assert Dim(2, 3, 4).r == Dim(4, 3, 2)


def test_Tensor():
    assert Tensor([1], Dim(1), Dim(1))
    m = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
    assert repr(m) == str(m)\
        == "Tensor[int64]([0, 1, 1, 0], dom=Dim(2), cod=Dim(2))"
    u = Tensor([1, 0, 0, 0], Dim(2), Dim(2))
    v = Tensor([0, 0, 0, 1], Dim(2), Dim(2))
    assert u + v == Tensor.id(Dim(2))
    with raises(TypeError):
        u + [0, 0, 0, 1]
    with raises(AxiomError):
        u + u @ Tensor([1, 0], Dim(1), Dim(2))
    with raises(TypeError):
        u >> Dim(2)
    arr = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape((2, 2, 2))
    m = Tensor(arr, Dim(2, 2), Dim(2))
    assert m == m and np.all(m.array == arr)
    m = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
    assert Tensor.id(Dim(2)).then(*(m, m)) == m >> m.dagger()


def test_Spider_to_tn():
    d = Dim(2)
    tensor = Spider(1, 1, d) >> Spider(1, 2, d) >> Spider(2, 0, d)
    result = tensor.eval(contractor=tn.contractors.auto).array
    assert all(result == np.array([1, 1]))


def test_Spider_to_tn_pytorch():
    try:
        with backend('pytorch') as np:
            tn.set_default_backend('pytorch')

            d = Dim(2)

            alice = Box("Alice", Dim(1), d,
                        np.array([1., 2.]).requires_grad_(True))
            tensor = alice >> Spider(1, 2, d) >> Spider(2, 0, d)
            result = tensor.eval(contractor=tn.contractors.auto).array
            assert result.item() == 3
    finally:
        tn.set_default_backend('numpy')


def test_Tensor_cups():
    assert np.all(Tensor.cups(Dim(2), Dim(2)).array == np.identity(2))
    with raises(TypeError):
        Tensor.cups(Dim(2), 2)
    with raises(TypeError):
        Tensor.cups(2, Dim(2))
    with raises(AxiomError):
        Tensor.cups(Dim(3), Dim(2))


def test_Tensor_caps():
    assert np.all(Tensor.caps(Dim(2), Dim(2)).array == np.identity(2))
    with raises(TypeError):
        Tensor.caps(Dim(2), 2)
    with raises(TypeError):
        Tensor.caps(2, Dim(2))
    with raises(AxiomError):
        Tensor.caps(Dim(3), Dim(2))


def test_Tensor_transpose():
    assert Tensor.caps(Dim(2), Dim(2)).transpose()\
        == Tensor.cups(Dim(2), Dim(2))


def test_Tensor_conjugate():
    assert Tensor[complex]([1j], Dim(1), Dim(1)).conjugate().array == -1j


def test_Tensor_tensor():
    assert Tensor.tensor(Tensor.id(Dim(2))) == Tensor.id(Dim(2))

    assert Tensor.id(Dim(2)) @ Tensor.id(Dim(3)) == Tensor.id(Dim(2, 3))

    v = Tensor([1, 0], Dim(1), Dim(2))
    assert v @ v == Tensor([1, 0, 0, 0], dom=Dim(1), cod=Dim(2, 2))
    assert v @ v.dagger() == v << v.dagger()

    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x, x), frobenius.Box('g', y, y)
    ob, ar = {x: 2, y: 3}, {f: [1, 0, 0, 1], g: list(range(9))}
    F = Functor(ob, ar)
    assert F(f) @ F(g) == F(f @ g)


def test_tensor_swap():
    f = Tensor([1, 0, 0, 1], Dim(2), Dim(2))
    g = Tensor(list(range(9)), Dim(3), Dim(3))
    swap = Tensor.swap(Dim(2), Dim(3))
    assert f @ g >> swap == swap >> g @ f


def test_tensor_spiders():
    with raises(NotImplementedError):
        Tensor.spiders(1, 2, Dim(3), [0.5])


def test_Functor_repr():
    x = frobenius.Ty('x')
    F = Functor({x: 2}, {}, dom=frobenius.Category(), dtype=bool)
    assert repr(F) ==\
        "tensor.Functor(ob={frobenius.Ty(frobenius.Ob('x')): 2}, ar={}, "\
        "dom=Category(frobenius.Ty, frobenius.Diagram), dtype=bool)"


def test_Functor_call():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x @ x, y), frobenius.Box('g', y, frobenius.Ty())
    ob = {x: 2, y: 3}
    ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
    F = Functor(ob, ar)
    assert list(F(f >> g).array.flatten()) == [5.0, 14.0, 23.0, 32.0]
    assert list(F(g.transpose(left=True)).array.flatten()) == [0.0, 1.0, 2.0]
    with raises(TypeError):
        F("Alice")
    assert Functor(ob={x: Dim(2, 3)}, ar=None)(x) == Dim(2, 3)


def test_Functor_swap():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f, g = frobenius.Box('f', x, x), frobenius.Box('g', y, y)
    F = Functor({x: 2, y: 3}, {f: [1, 2, 3, 4], g: list(range(9))})
    assert F(f @ g >> frobenius.Swap(x, y)) == \
           F(frobenius.Swap(x, y) >> g @ f)


def test_AxiomError():
    m = Tensor([1, 0, 0, 1, 0, 1, 1, 0], Dim(2, 2), Dim(2))
    with raises(AxiomError) as err:
        m >> m


def test_Functor_sum():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    f = frobenius.Box('f', x, y)
    F = Functor({x: 1, y: 2}, {f: [1, 0]})
    assert F(f + f) == F(f) + F(f)


def test_Tensor_radd():
    m = Tensor([1, 0, 0, 1, 0, 1, 1, 0], Dim(2, 2), Dim(2))
    assert 0 + m == m


def test_Tensor_iter():
    v = Tensor([0, 1], Dim(1), Dim(2))
    assert list(v) == [0, 1]
    s = Tensor([1], Dim(1), Dim(1))
    with raises(TypeError):
        # how does one iterate over a scalar?
        list(s)


def test_Tensor_subs():
    import sympy
    from sympy.abc import x
    s = Tensor[sympy.Expr]([x], Dim(1), Dim(1))
    assert s.subs(x, 1).array == 1


def test_Diagram_cups_and_caps():
    with raises(AxiomError):
        Diagram.cups(Dim(2), Dim(3))


def test_Diagram_swap():
    x, y, z = Dim(2), Dim(3), Dim(4)
    assert Diagram.swap(x, y @ z) == \
        (Swap(x, y) @ Id(z)) >> (Id(y) @ Swap(x, z))


def test_Box():
    f = Box('f', Dim(2), Dim(2), [0, 1, 1, 0])
    assert repr(f) == "tensor.Box('f', Dim(2), Dim(2), data=[0, 1, 1, 0])"
    assert {f: 42}[f] == 42


def test_Spider():
    assert repr(Spider(1, 2, Dim(3))) == "tensor.Spider(1, 2, Dim(3))"
    assert Spider(1, 2, Dim(2)).dagger() == Spider(2, 1, Dim(2))
    with raises(ValueError):
        Spider(1, 2, Dim(2, 3))


def test_Swap_to_tn():
    nodes, order = Swap(Dim(2), Dim(2)).to_tn()
    assert order == [nodes[0][0], nodes[1][0], nodes[1][1], nodes[0][1]]


def test_Tensor_scalar():
    s = Tensor([1], Dim(1), Dim(1))
    for ptype in [int, float, complex]:
        assert isinstance(ptype(s), ptype)


def test_Tensor_adjoint_eval():
    alice = Box("Alice", Dim(1), Dim(2), [1, 2])
    eats = Box("eats", Dim(1), Dim(2, 3, 2), [3] * 12)
    food = Box("food", Dim(1), Dim(2), [4, 5])

    diagram = alice @ eats @ food >>\
        Cup(Dim(2), Dim(2)) @ Id(Dim(3)) @ Cup(Dim(2), Dim(2))

    tensor1 = diagram.eval()
    tensor2 = diagram.transpose_box(2).transpose_box(0, left=True).eval()
    assert tensor1 == tensor2


def test_non_numpy_eval():
    with backend('pytorch'):
        with raises(Exception):
            Swap(Dim(2), Dim(2)).eval()


def test_Tensor_array():
    box = Box("box", Dim(2), Dim(2), None)
    assert box.array is None
