from pytest import raises

from discopy.interaction import *


def test_Ty_repr():
    t = Ty[int](positive=1, negative=2)
    assert repr(t)\
        == str(t) == "interaction.Ty[int](positive=1, negative=2)"


def test_Ty_str():
    x, y, z, w = map(Ty, "xyzw")
    assert str(x @ -y @ z @ -w) == "x @ z @ -y @ -w"


def test_Diagram_permutation():
    from discopy import compact
    x0, x1, y0, y1, z0, z1 = map(
        compact.Ty, ("x0", "x1", "y0", "y1", "z0", "z1"))
    x, y, z = (
        Ty[compact.Ty](x0, x1),
        Ty[compact.Ty](y0, y1),
        Ty[compact.Ty](z0, z1))
    diagram = Diagram[compact.Diagram]
    permutation = diagram.permutation([2, 0, 1], [x, y, z])
    assert permutation.dom == x @ y @ z
    assert permutation.cod == z @ x @ y
    assert diagram.permutation([0, 1, 2], [x, y, z])\
        == diagram.id(x @ y @ z)
    with raises(ValueError):
        diagram.permutation([1, 0], [x, y, z])


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
