# -*- coding: utf-8 -*-

from pytest import raises

from discopy import closed, compact, feedback, frobenius, markov
from discopy.para import (
    Closed, Compact, Feedback, Hypergraph, Markov, Symmetric, Traced)
from discopy.python import Function
from discopy.symmetric import Box, Diagram, Ty
from discopy.utils import AxiomError

x, y, z, p, q = map(Ty, "xyzpq")
f = Symmetric(x, y, Box('f', x @ p, y), p)
g = Symmetric(y, z, Box('g', y @ q, z), q)


def test_errors():
    with raises(AxiomError):
        Symmetric(x, y, Box('f', x @ p, y), q)
    with raises(AxiomError):
        Symmetric(x, z, Box('f', x @ p, y), p)
    with raises(AxiomError):
        g >> f
    with raises(AxiomError):
        f.reparam(Box('r', q, q))


def test_symmetric_axioms():
    assert f >> Symmetric.id(y) == f == Symmetric.id(x) >> f
    assert (f @ g.dom >> f.cod @ g).is_parallel(f @ g)
    swap = Symmetric.swap(x, y)
    assert (swap >> Symmetric.swap(y, x)).inside.simplify()\
        == Diagram.id(x @ y)
    assert Symmetric.permutation((1, 0), (x, y)).inside.simplify()\
        == swap.inside
    assert Symmetric.braid(x, y) == swap


def test_trace():
    t = Traced(x @ y, z @ y, Box('t', x @ y @ p, z @ y), p)
    assert t.trace(0) == t
    inside = x @ Diagram.swap(p, y) >> t.inside
    assert t.trace() == Traced(x, z, inside.trace(), p)
    u = Traced(y @ x, y @ z, Box('u', y @ x @ p, y @ z), p)
    assert u.trace(left=True) == Traced(x, z, u.inside.trace(left=True), p)


def test_reparam():
    r, s = Box('r', q, p), Box('s', p, q)
    assert f.reparam(r).param == q
    assert f.reparam(r).reparam(s) == f.reparam(s >> r)
    assert (f @ g).reparam(r @ s).param == q @ p


def test_markov():
    X = markov.Ty('x')
    assert Markov.copy(X, 3) == Markov.lift(markov.Diagram.copy(X, 3))
    assert Markov.copy(X).param == markov.Ty()


def test_closed():
    a, b, c, P = map(closed.Ty, "abcP")
    k = Closed(a @ b, c, closed.Box('k', a @ b @ P, c), P)
    assert Closed.ev(c, b).param == closed.Ty()
    left, right = k.curry(left=True), k.curry(left=False)
    assert (left.dom, left.cod, left.param) == (a, c << b, P)
    assert (right.dom, right.cod, right.param) == (b, a >> c, P)
    assert left.inside\
        == (a @ closed.Diagram.swap(P, b) >> k.inside).curry(left=True)


def test_feedback():
    x, y, z, P = map(feedback.Ty, "xyzP")
    f = Feedback(x @ y.delay(), z @ y,
                 feedback.Box('f', x @ y.delay() @ P, z @ y), P)
    fb = f.feedback()
    assert (fb.dom, fb.cod, fb.param) == (x, z, P)
    assert f.delay().param == P.delay()


def test_compact():
    x, y = map(compact.Ty, "xy")
    assert Compact.cups(x, x.r) == Compact.lift(compact.Diagram.cups(x, x.r))
    assert Compact.caps(x, x.l).param == compact.Ty()
    k = Compact(x @ y, x,
                compact.Box('k', x @ y @ compact.Ty('P'), x),
                compact.Ty('P'))
    assert k.curry(left=True).cod == x << y


def test_hypergraph():
    X = frobenius.Ty('x')
    assert Hypergraph.spiders(1, 2, X)\
        == Hypergraph.lift(frobenius.Diagram.spiders(1, 2, X))


def test_python():
    inside = Function(lambda a, w, b: w * a + b, (float, ) * 3, (float, ))
    layer = Symmetric[Function](
        (float, ), (float, ), inside, (float, float))
    pair = layer @ layer
    assert pair.inside(1., 2., 10., 0., 3., 5.) == (10., 11.)


def test_copar():
    m, n = Ty('m'), Ty('n')
    t = Symmetric(x, y, Box('t', x @ p, y @ m), p, m)
    with raises(AxiomError):
        Symmetric(x, y, Box('t', x @ p, y @ m), p, n)
    with raises(AxiomError):
        Symmetric(x, y, Box('t', x @ p, y @ m), p)
    with raises(AxiomError):
        t.recopar(Box('c', n, n))
    assert t >> Symmetric.id(y) == t == Symmetric.id(x) >> t
    assert t.reparam(Box('r', q, p)).copar == m
    assert t.recopar(Box('c', m, n)).param == p
    assert (t @ t).param == p @ p and (t @ t).copar == m @ m


def test_copar_trace():
    m = Ty('m')
    t = Traced(x @ y, z @ y, Box('t', x @ y @ p, z @ y @ m), p, m)
    trace = t.trace()
    assert (trace.dom, trace.cod, trace.param, trace.copar) == (x, z, p, m)
    u = Traced(y @ x, y @ z, Box('u', y @ x @ p, y @ z @ m), p, m)
    assert u.trace(left=True).copar == m


def test_copar_feedback():
    x, y, z, m, P = map(feedback.Ty, "xyzmP")
    f = Feedback(x @ y.delay(), z @ y,
                 feedback.Box('f', x @ y.delay() @ P, z @ y @ m), P, m)
    fb = f.feedback()
    assert (fb.dom, fb.cod, fb.param, fb.copar) == (x, z, P, m)
    assert f.delay().copar == m.delay()


def test_copar_python():
    add = Function(lambda a, s: (a + s, a), (float, ) * 2, (float, ) * 2)
    cell = Symmetric[Function](
        (float, ), (float, ), add, (float, ), (float, ))
    network = cell >> cell
    assert network.param == network.copar == (float, float)
    assert network.inside(2., 1., 10.) == (13., 2., 3.)
