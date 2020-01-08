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
