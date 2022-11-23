from pytest import raises
from discopy.monoidal import Encoding
from discopy.compact import *


def test_Cup_Cap_dagger():
    n = Ty('n')
    assert Cap(n, n.l).dagger() == Cup(n, n.l)
    assert Cup(n, n.l).dagger() == Cap(n, n.l)


def test_transpose_box():
    n = Ty('s')
    Bob = Box('Bob', Ty(), n)
    Bob_Tl = Box('Bob', n.l, Ty(), _z=-1, is_dagger=True)
    Bob_Tr = Box('Bob', n.r, Ty(), _z=1, is_dagger=True)
    assert Bob.transpose_box(0, left=True) == Cap(n.r, n) >> Bob_Tr @ Id(n)
    assert Bob.transpose_box(0) == Cap(n, n.l) >> Id(n) @ Bob_Tl


def test_cup_chaining():
    n, s, p = map(Ty, "nsp")
    A = Box('A', Ty(), n @ p)
    V = Box('V', Ty(), n.r @ s @ n.l)
    B = Box('B', Ty(), p.r @ n)

    diagram = (A @ V @ B).cup(1, 5).cup(0, 1).cup(1, 2)
    expected_boxes = [
        A, V, B, Swap(p, n.r), Swap(p, s), Swap(p, n.l), Cup(p, p.r),
        Cup(n, n.r), Cup(n.l, n)]
    expected_offsets = [0, 2, 5, 1, 2, 3, 4, 0, 1]
    expected_diagram = Diagram.decode(Encoding(
        dom=Ty(), boxes_and_offsets=tuple(zip(
            expected_boxes, expected_offsets))))
    assert diagram == expected_diagram

    with raises(ValueError):
        Id(n @ n.r).cup(0, 2)
