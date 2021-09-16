from pytest import raises
from unittest.mock import Mock
import numpy as np
from discopy.tensor import *


def test_Dim():
    with raises(TypeError):
        Dim('a')
    with raises(ValueError):
        Dim(-1)
    dim = Dim(2, 3)
    assert Dim(1) @ dim == dim @ Dim(1) == dim
    assert Dim(1).tensor(*(Dim(2, 3), Dim(4), Dim(1))) == Dim(2, 3, 4)
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
    assert u + v == Tensor.id(Dim(2))
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
    assert Tensor.id(Dim(2)).then(*(m, m)) == m >> m.dagger()


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


def test_Tensor_tensor():
    assert Tensor.tensor(Tensor.id(Dim(2))) == Tensor.id(Dim(2))

    assert Tensor.id(Dim(2)) @ Tensor.id(Dim(3)) == Tensor.id(Dim(2, 3))

    v = Tensor(Dim(1), Dim(2), [1, 0])
    assert v @ v == Tensor(dom=Dim(1), cod=Dim(2, 2), array=[1, 0, 0, 0])
    assert v @ v.dagger() == v << v.dagger()

    x, y = Ty('x'), Ty('y')
    f, g = rigid.Box('f', x, x), rigid.Box('g', y, y)
    ob, ar = {x: 2, y: 3}, {f: [1, 0, 0, 1], g: list(range(9))}
    F = Functor(ob, ar)
    assert F(f) @ F(g) == F(f @ g)


def test_tensor_swap():
    f = Tensor(Dim(2), Dim(2), [1, 0, 0, 1])
    g = Tensor(Dim(3), Dim(3), list(range(9)))
    swap = Tensor.swap(Dim(2), Dim(3))
    assert f @ g >> swap == swap >> g @ f


def test_Functor():
    assert repr(Functor({Ty('x'): 1}, {})) ==\
        "tensor.Functor(ob={Ty('x'): 1}, ar={})"


def test_Functor_call():
    x, y = Ty('x'), Ty('y')
    f, g = rigid.Box('f', x @ x, y), rigid.Box('g', y, Ty())
    ob = {x: 2, y: 3}
    ar = {f: list(range(2 * 2 * 3)), g: list(range(3))}
    F = Functor(ob, ar)
    assert list(F(f >> g).array.flatten()) == [5.0, 14.0, 23.0, 32.0]
    assert list(F(g.transpose(left=True)).array.flatten()) == [0.0, 1.0, 2.0]
    with raises(TypeError):
        F("Alice")
    assert Functor(ob={x: Ty(2, 3)}, ar=None)(x) == Dim(2, 3)


def test_Functor_swap():
    x, y = Ty('x'), Ty('y')
    f, g = rigid.Box('f', x, x), rigid.Box('g', y, y)
    F = Functor({x: 2, y: 3}, {f: [1, 2, 3, 4], g: list(range(9))})
    assert F(f @ g >> rigid.Diagram.swap(x, y)) == \
           F(rigid.Diagram.swap(x, y) >> g @ f)


def test_AxiomError():
    m = Tensor(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    with raises(AxiomError) as err:
        m >> m
    assert str(err.value) == messages.does_not_compose(m, m)


def test_Functor_sum():
    x, y = Ty('x'), Ty('y')
    f = rigid.Box('f', x, y)
    F = Functor({x: 1, y: 2}, {f: [1, 0]})
    assert F(f + f) == F(f) + F(f)


def test_Tensor_radd():
    m = Tensor(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    assert 0 + m == m


def test_Tensor_iter():
    v = Tensor(Dim(1), Dim(2), [0, 1])
    assert list(v) == [0, 1]
    s = Tensor(Dim(1), Dim(1), [1])
    assert list(s) == [1]


def test_Tensor_subs():
    from sympy.abc import x
    s = Tensor(Dim(1), Dim(1), [x])
    assert s.subs(x, 1) == 1


def test_Diagram_cups_and_caps():
    with raises(AxiomError):
        Diagram.cups(Dim(2), Dim(3))
    assert Id(Dim(2)).transpose()\
        == Spider(0, 2, dim=2) @ Id(Dim(2))\
        >> Id(Dim(2)) @ Spider(2, 0, dim=2)


def test_Diagram_swap():
    x, y, z = Dim(2), Dim(3), Dim(4)
    assert Diagram.swap(x, y @ z) == \
           (Swap(x, y) @ Id(z)) >> (Id(y) @ Swap(x, z))


def test_Box():
    f = Box('f', Dim(2), Dim(2), [0, 1, 1, 0])
    assert repr(f) == "tensor.Box('f', Dim(2), Dim(2), data=[0, 1, 1, 0])"
    assert f != rigid.Box('f', Dim(2), Dim(2), data=[0, 1, 1, 0])
    assert {f: 42}[f] == 42


def test_Spider():
    assert repr(Spider(1, 2, dim=3)) == "Spider(1, 2, Dim(3))"
    assert Spider(1, 2, 2).dagger() == Spider(2, 1, 2)
    with raises(ValueError):
        Spider(1, 2, Dim(2, 3))


def test_Swap_to_tn():
    nodes, order = Swap(Dim(2), Dim(2)).to_tn()
    assert order == [nodes[0][0], nodes[1][0], nodes[1][1], nodes[0][1]]


def test_Tensor_scalar():
    s = Tensor(Dim(1), Dim(1), [1])
    for ptype in [int, float, complex]:
        assert isinstance(ptype(s), ptype)


def test_Tensor_adjoint_functor():
    from discopy.rigid import Ty, Cup
    from discopy.grammar import Word

    n, s = map(Ty, 'ns')
    alice = Word("Alice", n)
    eats = Word("eats", n.r @ s @ n.l)
    food = Word("food", n)

    diagram = alice @ eats @ food >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)

    f = Functor(
        ob={n: Dim(2), s: Dim(3)},
        ar={
            alice: Tensor(Dim(1), Dim(2), [1, 2]),
            eats: Tensor(Dim(1), Dim(2, 3, 2), [3] * 12),
            food: Tensor(Dim(1), Dim(2), [4, 5])
        })

    tensor1 = f(diagram)
    tensor2 = f(diagram.transpose_box(2).transpose_box(0))
    assert tensor1 == tensor2


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
    Tensor.np = Mock(__package__='pytorch')
    with raises(Exception):
        Swap(Dim(2), Dim(2)).eval()
    Tensor.np = np
