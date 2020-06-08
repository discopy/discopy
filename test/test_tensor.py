from pytest import raises
from discopy.tensor import *


def test_Dim():
    with raises(TypeError):
        Dim('a')
    with raises(ValueError):
        Dim(-1)
    dim = Dim(2, 3)
    assert Dim(1) @ dim == dim @ Dim(1) == dim
    assert sum([Dim(1), dim, Dim(4)], Dim(1)) == Dim(2, 3, 4)
    assert dim[:1] == Dim(3, 2)[1:] == Dim(2)
    assert dim[0] == Dim(3, 2)[1] == 2
    assert repr(Dim(1, 2, 3)) == str(dim) == "Dim(2, 3)"
    assert {dim: 42}[dim] == 42
    assert Dim(2, 3, 4).r == Dim(4, 3, 2)


def test_Tensor():
    assert Tensor(Dim(1), Dim(1), [1])
    m = Tensor(Dim(2), Dim(2), [0, 1, 1, 0])
    assert repr(m) == str(m)\
        == "Tensor(dom=Dim(2), cod=Dim(2), array=[0, 1, 1, 0])"
    u = Tensor(Dim(2), Dim(2), [1, 0, 0, 0])
    v = Tensor(Dim(2), Dim(2), [0, 0, 0, 1])
    assert u + v == Id(2)
    with raises(TypeError):
        u + [0, 0, 0, 1]
    with raises(AxiomError):
        u + u @ Tensor(Dim(1), Dim(2), [1, 0])
    with raises(TypeError):
        u >> Dim(2)
    with raises(TypeError):
        u @ Dim(2)
    arr = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape((2, 2, 2))
    m = Tensor(Dim(2, 2), Dim(2), arr)
    assert m == m and np.all(m == arr)
    m = Tensor(Dim(2), Dim(2), [0, 1, 1, 0])
    assert m >> m == m >> m.dagger() == Id(2)


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


def test_Tensor_tensor():
    v = Tensor(Dim(1), Dim(2), [1, 0])
    assert v @ v == Tensor(dom=Dim(1), cod=Dim(2, 2), array=[1, 0, 0, 0])
    assert v @ v.dagger() == v << v.dagger()


def test_TensorFunctor():
    assert repr(TensorFunctor({Ty('x'): 1}, {})) ==\
        "TensorFunctor(ob={Ty('x'): 1}, ar={})"


def test_TensorFunctor_call():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x @ x, y), Box('g', y, Ty())
    ob = {x: 2, y: 3}
    ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
    F = TensorFunctor(ob, ar)
    assert list(F(f >> g).array.flatten()) == [5.0, 14.0, 23.0, 32.0]
    assert list(F(g.transpose_l()).array.flatten()) == [0.0, 1.0, 2.0]
    with raises(TypeError):
        F("Alice")


def test_AxiomError():
    m = Tensor(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    with raises(AxiomError) as err:
        m >> m
    assert str(err.value) == messages.does_not_compose(m, m)
