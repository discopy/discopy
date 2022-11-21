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
