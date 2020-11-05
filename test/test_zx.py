from pytest import raises
from discopy import *
from discopy.zx import *


def test_Diagram():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    repr(bialgebra) == "zx.Diagram(dom=PRO(2), cod=PRO(2), "\
                       "boxes=[Z(1, 2), Z(1, 2), SWAP, X(2, 1), X(2, 1)], "\
                       "offsets=[0, 2, 1, 0, 1])"
    str(bialgebra) == "Z(1, 2) @ Id(1) >> Id(2) @ Z(1, 2)"\
                      ">> Id(1) @ SWAP @ Id(1)"\
                      ">> X(2, 1) @ Id(2) >> Id(1) @ X(2, 1)"

def test_Swap():
    x = Ty('x')
    with raises(TypeError):
        Swap(x, x)
    with raises(TypeError):
        Swap(PRO(1), x)


def test_Spider():
    assert repr(Z(1, 2, 3)) == "Z(1, 2, 3)"
    assert Z(1, 2, 3).phase == 3


def test_Sum():
    assert Z(1, 1) + Z(1, 1) >> Z(1, 1) == sum(2 * [Z(1, 1) >> Z(1, 1)])


def test_Functor():
    x = Ty('x')
    f = Box('f', x, x)
    F = rigid.Functor(
        ob=lambda _: PRO(1),
        ar=lambda f: Z(len(f.dom), len(f.cod)),
        ob_factory=PRO,
        ar_factory=Diagram)
    assert F(f) == Z(1, 1)
    assert F(rigid.Swap(x, x)) == Diagram.permutation([1, 0]) == SWAP
    assert F(Cup(x.l, x)) == Z(2, 0)
    assert F(Cap(x.r, x)) == Z(0, 2)
    assert F(f + f) == Z(1, 1) + Z(1, 1)
