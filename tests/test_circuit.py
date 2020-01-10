from pytest import raises
from discopy.circuit import *


def test_PRO_r():
    assert PRO(2).r == PRO(2)


def test_PRO_tensor():
    assert PRO(2) @ PRO(3) @ PRO(7) == PRO(12)


def test_PRO_init():
    assert list(PRO(0)) == []
    assert all(len(PRO(n)) == n for n in range(5))


def test_PRO_repr():
    assert repr((PRO(0), PRO(1))) == "(PRO(0), PRO(1))"


def test_PRO_str():
    assert str(PRO(2 * 3 * 7)) == "PRO(42)"


def test_PRO_getitem():
    assert PRO(42)[2: 4] == PRO(2)
    assert all(PRO(42)[i].name == 1 for i in range(42))


def test_Circuit_cups():
    with raises(TypeError):
        Circuit.cups(2, PRO(3))
    with raises(TypeError):
        Circuit.cups(PRO(2), 3)


def test_Circuit_from_tk():
    with raises(NotImplementedError):
        Circuit.from_tk(Id(3).to_tk().CCX(0, 1, 2))
    circuit = Circuit.from_tk(Id(3).to_tk().CX(0, 2))
    assert circuit == Id(1) @ SWAP >> CX @ Id(1)
    circuit = Circuit.from_tk(Id(3).to_tk().CX(2, 0))
    assert circuit == SWAP @ Id(1) >> Id(1) @ SWAP >> Id(1) @ CX
