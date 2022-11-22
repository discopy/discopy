from discopy.tortile import *


def test_Kauffman():
    tmp = Ty.l, Ty.r
    Ty.l = Ty.r = property(lambda self: self)
    x, A = Ty('x'), Box('A', Ty(), Ty())

    class Polynomial(Diagram):
        def braid(x, y):
            return (A @ x @ y) + (Cup(x, y) >> A.dagger() >> Cap(x, y))

    Kauffman = Functor(ob={x: x}, ar={}, cod=Category(Ty, Polynomial))

    assert Kauffman(Braid(x, x))\
        == (A @ x @ x) + (Cup(x, x) >> A.dagger() >> Cap(x, x))

    Ty.l, Ty.r = tmp
