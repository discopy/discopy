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


def test_fixed_point():
    phi = Function(lambda x=1: 1 + 1 / x, [int], [int]).fix()
    assert phi() == (1 + sqrt(5)) / 2
