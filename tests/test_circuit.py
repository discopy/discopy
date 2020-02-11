from pytest import raises
from discopy.circuit import *


def test_Circuit_cups():
    with raises(TypeError):
        Circuit.cups(2, PRO(3))
    with raises(TypeError):
        Circuit.cups(PRO(2), 3)


def test_Circuit_to_tk():
    bell_state = Circuit.caps(PRO(1), PRO(1))
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Id(1) @ bell_effect)[::-1]
    tk_circ = snake.to_tk()
    assert abs(tk_circ.scalar - 2) < 1e-5
    assert tk_circ.post_selection == {1: 0, 2: 0}
    assert list(map(str, tk_circ)) == [
        "H q[2];",
        "CX q[2], q[0];",
        "CX q[1], q[2];",
        "H q[1];"]
    assert Circuit.from_tk(snake.to_tk()) ==\
        Id(2) @ H\
        >> SWAP @ Id(1)\
        >> Id(1) @ SWAP\
        >> Id(1) @ CX\
        >> Id(1) @ SWAP\
        >> SWAP @ Id(1)\
        >> Id(1) @ CX\
        >> Id(1) @ H @ Id(1)\
        >> Id(2) @ Bra(0)\
        >> Id(1) @ Bra(0)\
        >> Id(1) @ scalar(2.000)


def test_Circuit_from_tk():
    with raises(TypeError):
        Circuit.from_tk(CX)
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
