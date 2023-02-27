from pytest import raises

from discopy.interaction import *


def test_Ty_repr():
    assert repr(Ty[int](positive=1, negative=2))\
            == "interaction.Ty[builtins.int](positive=1, negative=2)"
    x, y, z, w = map(Ty, "xyzw")
    assert str(x @ -y @ z @ -w) == "x @ z @ -(w @ y)"


def test_ValueError():
    from discopy.ribbon import Ty as T, Diagram as D, Box as B
    x, y, z = map(Ty[T], "xyz")
    f = B('f', T('x'), T('y'))
    with raises(ValueError):
        Diagram[D](f, x, z)
    with raises(ValueError):
        Diagram[D](f, z, y)
