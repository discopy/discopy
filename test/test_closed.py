# -*- coding: utf-8 -*-

from discopy.python import *
from discopy.closed import *


def test_Function():
    x, y, z = (complex, ), (bool, ), (float, )
    f = Function(dom=y, cod=exp(z, x),
                 inside=lambda y: lambda x: abs(x) ** 2 if y else 0)
    g = Function(dom=x + y, cod=z, inside=lambda x, y: f(y)(x))

    assert f.uncurry().curry()(True)(1j) == f(True)(1j)
    assert f.uncurry(left=False).curry(left=False)(True)(1j) == f(True)(1j)
    assert g.curry().uncurry()(True, 1j) == g(True, 1j)
    assert g.curry(left=False).uncurry(left=False)(True, 1j) == g(True, 1j)


def test_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', y, z << x), Box('g', y, z >> x)

    F = Functor(
        ob={x: complex, y: bool, z: float},
        ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda y: lambda z: z + 1j if y else -1j},
        cod=Category(tuple[type, ...], Function))

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.uncurry(left=False).curry(left=False))(True)(1.2) == F(g)(True)(1.2)


def test_drawing():
    from discopy import rigid

    @factory
    class ClosedDrawing(rigid.Diagram):
        eval = staticmethod(lambda base, exponent, left=True:
            rigid.Diagram.eval(base, exponent, left).bubble())
        curry = lambda self, n=1, left=True:\
            rigid.Diagram.curry(self, n, left).bubble()

    class DrawingBox(rigid.Box, ClosedDrawing):
        pass

    Draw = Functor(
        lambda x: rigid.Ty(x.name),
        lambda f: DrawingBox(f.name, Draw(f.dom), Draw(f.cod)),
        cod=Category(rigid.Ty, ClosedDrawing))
    Diagram.draw = lambda self, **params: Draw(self)

    x, y, z = map(Ty, "xyz")
    f, g, h = Box('f', x, z << y), Box('g', x @ y, z), Box('h', y, x >> z)
    f.uncurry().draw()
    f.uncurry().curry().draw()
    h.uncurry(left=False).curry(left=False).draw()
    g.curry().uncurry().draw(), g, g.curry(left=False).uncurry(left=False).draw()
