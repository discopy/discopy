from __future__ import annotations

from discopy.closed import *


def test_exp():
    X, Y = Ty('X'), Ty('Y')
    assert X >> Y == Y ** X == Y << X
    assert X @ Ty() == X == Ty() @ X


def test_str():
    X, Y = Ty("X"), Ty("Y")
    f = X(lambda x: (X >> Y)(lambda y: y(x)))
    assert str(f) == "X(lambda x: (X >> Y)(lambda y: y(x)))"


def test_python_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', y, x >> z), Box('g', x @ y, z)

    from discopy.python import Function
    F = Functor(
        ob={x: complex, y: bool, z: float},
        ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j)},
        cod=Function)

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


def test_to_closed():
    from discopy import biclosed

    x, y = biclosed.Ty('x'), biclosed.Ty('y')
    X, Y = Ty('x'), Ty('y')

    # Left and right exponentials collapse to the same closed type.
    over, under = biclosed.Box('f', x, x << y), biclosed.Box('f', x, y >> x)
    assert over.to_closed() == under.to_closed() == Box('f', X, X ** Y)

    # Evaluation is preserved, keeping the side of the argument.
    assert biclosed.Diagram.ev(x, y, left=True).to_closed()\
        == Diagram.ev(X, Y, left=True)
    assert biclosed.Diagram.ev(x, y, left=False).to_closed()\
        == Diagram.ev(X, Y, left=False)


def test_fx_bx():
    x, y, z = map(Ty, "xyz")

    fx = Diagram.fx(x, y, z)
    assert fx.dom == (x ** y) @ (y ** z) and fx.cod == z >> x

    bx = Diagram.bx(x, y, z)
    assert bx.dom == (y ** x) @ (z ** y) and bx.cod == z << x
