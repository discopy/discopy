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
    assert str(PRO(2 * 3 * 7)) == "42"


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
    circuit = Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
    tk_circ = circuit.to_tk()
    circuit1 = Ket(0, 0, 0) >> Circuit.from_tk(tk_circ)
    assert np.all(circuit.eval() == circuit1.eval())


def test_Circuit_normal_form():
    caps = Circuit.caps(PRO(2), PRO(2))
    cups = Circuit.cups(PRO(2), PRO(2))
    snake = caps @ Id(2) >> Id(2) @ cups
    circ = snake.normal_form()
    assert circ.boxes[0] == Ket(0, 0, 0, 0)
    assert circ.boxes[-2] == Bra(0, 0, 0, 0)
    assert circ.boxes[-1].name == '4.000'
