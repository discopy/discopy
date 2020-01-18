from pytest import raises
from discopy.matrix import *


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


def test_Matrix():
    assert Matrix(Dim(1), Dim(1), [1])
    m = Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
    assert repr(m) == str(m)\
        == "Matrix(dom=Dim(2), cod=Dim(2), array=[0, 1, 1, 0])"
    u = Matrix(Dim(2), Dim(2), [1, 0, 0, 0])
    v = Matrix(Dim(2), Dim(2), [0, 0, 0, 1])
    assert u + v == Id(2)
    with raises(TypeError):
        u + [0, 0, 0, 1]
    with raises(AxiomError):
        u + u @ Matrix(Dim(1), Dim(2), [1, 0])
    with raises(TypeError):
        u >> Dim(2)
    with raises(TypeError):
        u @ Dim(2)
    arr = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape((2, 2, 2))
    m = Matrix(Dim(2, 2), Dim(2), arr)
    assert m == m and np.all(m == arr)
    m = Matrix(Dim(2), Dim(2), [0, 1, 1, 0])
    assert m >> m == m >> m.dagger() == Id(2)


def test_Matrix_cups():
    assert np.all(Matrix.cups(Dim(2), Dim(2)).array == np.identity(2))
    with raises(TypeError):
        Matrix.cups(Dim(2), 2)
    with raises(TypeError):
        Matrix.cups(2, Dim(2))
    with raises(AxiomError):
        Matrix.cups(Dim(3), Dim(2))


def test_Matrix_caps():
    assert np.all(Matrix.caps(Dim(2), Dim(2)).array == np.identity(2))
    with raises(TypeError):
        Matrix.caps(Dim(2), 2)
    with raises(TypeError):
        Matrix.caps(2, Dim(2))
    with raises(AxiomError):
        Matrix.caps(Dim(3), Dim(2))


def test_Matrix_tensor():
    v = Matrix(Dim(1), Dim(2), [1, 0])
    assert v @ v == Matrix(dom=Dim(1), cod=Dim(2, 2), array=[1, 0, 0, 0])
    assert v @ v.dagger() == v << v.dagger()


def test_MatrixFunctor():
    assert repr(MatrixFunctor({Ty('x'): 1}, {})) ==\
        "MatrixFunctor(ob={Ty('x'): 1}, ar={})"


def test_AxiomError():
    m = Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    with raises(AxiomError) as err:
        m >> m
    assert str(err.value) == messages.does_not_compose(m, m)
