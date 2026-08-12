# -*- coding: utf-8 -*-

from pytest import raises

from discopy.para import Para, Reparam
from discopy.python import Function
from discopy.symmetric import Box, Diagram, Ty
from discopy.utils import AxiomError

x, y, z, p, q = map(Ty, "xyzpq")
f = Para(x, y, p, Box('f', x @ p, y))
g = Para(y, z, q, Box('g', y @ q, z))


def test_errors():
    with raises(AxiomError):
        Para(x, y, q, Box('f', x @ p, y))
    with raises(AxiomError):
        Para(x, z, p, Box('f', x @ p, y))
    with raises(AxiomError):
        g >> f
    with raises(AxiomError):
        f.reparam(Box('r', q, q))


def test_symmetric_axioms():
    assert f >> Para.id(y) == f == Para.id(x) >> f
    assert (f @ g.dom >> f.cod @ g).is_parallel(f @ g)
    swap = Para.swap(x, y)
    assert (swap >> Para.swap(y, x)).inside.simplify() == Diagram.id(x @ y)
    assert Para.permutation((1, 0), (x, y)).inside.simplify()\
        == swap.inside
    assert Para.braid(x, y) == swap and Para.twist(x) == Para.id(x)


def test_trace():
    t = Para(x @ y, z @ y, p, Box('t', x @ y @ p, z @ y))
    assert t.trace(0) == t
    inside = x @ Diagram.swap(p, y) >> t.inside
    assert t.trace() == Para(x, z, p, inside.trace())
    u = Para(y @ x, y @ z, p, Box('u', y @ x @ p, y @ z))
    assert u.trace(left=True) == Para(x, z, p, u.inside.trace(left=True))


def test_reparam():
    r, s = Box('r', q, p), Box('s', p, q)
    assert Reparam.id(f) >> Reparam(f, f.reparam(r), r)\
        == Reparam(f, f.reparam(r), r)
    with raises(AxiomError):
        Reparam(f, g, Box('r', q, p))
    with raises(AxiomError):
        Reparam(f, f.reparam(r), Box('r', q, q))
    with raises(AxiomError):
        Reparam(f, f.reparam(r), r) >> Reparam.id(f)
    alpha, beta = Reparam(f, f.reparam(r), r), Reparam(g, g.reparam(s), s)
    assert (alpha @ beta).source == f @ g
    assert (alpha @ beta).target == f.reparam(r) @ g.reparam(s)
    assert (alpha @ beta).inside == r @ s
    assert (alpha @ g).inside == r @ q and (f @ beta).inside == p @ s


def test_python():
    inside = Function(lambda a, w, b: w * a + b, (float, ) * 3, (float, ))
    layer = Para[Function]((float, ), (float, ), (float, float), inside)
    network = layer >> layer
    assert network.dom == network.cod == (float, )
    assert network.param == (float, ) * 4
    assert network.inside(2., 3., 1., .5, 0.) == 3.5
    pair = layer @ layer
    assert pair.inside(1., 2., 10., 0., 3., 5.) == (10., 11.)
