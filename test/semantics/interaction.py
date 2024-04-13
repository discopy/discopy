from pytest import raises

from discopy.interaction import *


def test_Ty_repr():
    t = Ty[int](positive=1, negative=2)
    assert repr(t)\
        == str(t) == "interaction.Ty[int](positive=1, negative=2)"


def test_Ty_str():
    x, y, z, w = map(Ty, "xyzw")
    assert str(x @ -y @ z @ -w) == "x @ z @ -y @ -w"


def test_ValueError():
    from discopy.ribbon import Ty as T, Diagram as D, Box as B
    x, y, z = map(Ty[T], "xyz")
    f = B('f', T('x'), T('y'))
    with raises(ValueError):
        Diagram[D](f, x, z)
    with raises(ValueError):
        Diagram[D](f, z, y)


def test_IndexError():
    with raises(IndexError):
        return Id()[:]
