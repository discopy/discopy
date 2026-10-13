# -*- coding: utf-8 -*-

import random

from pytest import raises

from discopy import cat, lens, markov, monoidal, para, python, symmetric
from discopy.abc import Nat
from discopy.interaction import Ty as IntTy
from discopy.lens import Markov, Symmetric, Traced, Ty
from discopy.python import Function
from discopy.symmetric import Box, Diagram
from discopy.symmetric import Ty as T
from discopy.utils import AxiomError

x, x_, y, y_, z, z_, w, w_ = map(
    T, ["x", "x'", "y", "y'", "z", "z'", "w", "w'"])
m, n, k = map(T, "mnk")
X, Y, Z, W = Ty(x, x_), Ty(y, y_), Ty(z, z_), Ty(w, w_)
f = Symmetric(X, Y, Box('f', x, y @ m), Box("f'", m @ y_, x_), m)
g = Symmetric(Y, Z, Box('g', y, z @ n), Box("g'", n @ z_, y_), n)
h = Symmetric(Z, W, Box('h', z, w @ k), Box("h'", k @ w_, z_), k)


def equal(left, right):
    """ Equality of lenses up to sliding, decided by the hypergraph. """
    return (left.dom, left.cod) == (right.dom, right.cod)\
        and left.to_int().inside.to_hypergraph()\
        == right.to_int().inside.to_hypergraph()


def traced():
    return Traced(X, Y, f.forward, f.backward, m)


def test_ty():
    assert Ty[Nat](Nat(1), Nat(2)) @ Ty[Nat](Nat(3), Nat(4)) == Ty[Nat](
        Nat(4), Nat(6))
    assert -(X @ Y) == -X @ -Y == Ty(x_ @ y_, x @ y)
    scope = {"cat": cat, "monoidal": monoidal, "lens": lens,
             "symmetric": symmetric, "markov": markov}
    assert eval(repr(X), scope) == X and str(X) == "x @ -x'"
    assert eval(repr(f), scope) == f
    assert eval(repr(markov_lens()), scope) == markov_lens()
    assert eval(repr(traced()), scope) == traced() != f
    assert repr(square).startswith(
        "lens.Markov[python.multiplicative.Function](")
    assert repr(Symmetric[Function].id(R)).startswith(
        "lens.Symmetric[python.multiplicative.Function](")
    assert Ty[Nat]() == Ty[Nat](Nat(0), Nat(0)) and Ty.unit() == Ty()
    assert Ty[Nat].unit() == Ty[Nat]() \
        and Ty[python.Ty].unit() == Ty[python.Ty]()
    assert Ty.negatives is tuple and IntTy.negatives is reversed


def test_errors():
    with raises(TypeError):
        Symmetric(x, y, Box('f', x, y), Box("f'", y_, x_))
    with raises(TypeError):
        Symmetric(IntTy[T](x, x_), Y, f.forward, f.backward, m)
    with raises(TypeError):
        Symmetric(X, Y, f.forward, f.backward, "m")
    with raises(TypeError):
        f >> Markov.id(Y)
    with raises(TypeError):
        f @ Markov.id(Y)
    with raises(TypeError):
        markov_lens() >> Symmetric.id(Y)
    with raises(AxiomError):
        Symmetric(X, Y, Box('f', x, y @ m), Box("f'", m @ y_, x_), n)
    with raises(AxiomError):
        Symmetric(X, Z, Box('f', x, y @ m), Box("f'", m @ y_, x_), m)
    with raises(AxiomError):
        g >> f
    with raises(AxiomError):
        Markov(X, Y, markov.Box('get', x, y), markov.Box('put', x @ y_, x))


def test_category_axioms():
    assert f >> Symmetric.id(Y) == f == Symmetric.id(X) >> f
    assert equal((f >> g) >> h, f >> (g >> h))
    assert f >> g >> h == Symmetric.then(f, g, h)
    assert (f @ h).dom == X @ Z and (f @ h).cod == Y @ W
    assert (f @ h).residual == m @ k
    assert equal(f @ h.dom >> f.cod @ h, f @ h)
    assert equal((f @ g) >> (g @ h), (f >> g) @ (g >> h))
    assert equal((f @ h) @ g, f @ (h @ g))
    assert ((f @ h) @ g).residual == (f @ (h @ g)).residual
    assert f @ Symmetric.id() == f == Symmetric.id() @ f
    assert Symmetric.id() == Symmetric.lift(Diagram.id(T()))


def test_lift():
    forward, backward = Box('p', x, y), Box('q', y_, x_)
    assert Symmetric.lift(forward, backward)\
        == Symmetric(X, Y, forward, backward)
    assert Symmetric.lift(forward) == Symmetric(
        Ty(x, T()), Ty(y, T()), forward, Diagram.id(T()))
    get, put = markov.Box('p', x, y), markov.Box('q', y_, x_)
    assert Markov.lift(get, put).put == markov.Diagram.discard(x) @ put


def test_symmetric_axioms():
    swap = Symmetric.swap(X, Y)
    assert swap.dom == X @ Y and swap.cod == Y @ X
    assert (swap >> Symmetric.swap(Y, X)).forward.simplify()\
        == Diagram.id(x @ y)
    assert (swap >> Symmetric.swap(Y, X)).backward.simplify()\
        == Diagram.id(x_ @ y_)
    assert Symmetric.permutation((1, 0), (X, Y)).forward.simplify()\
        == swap.forward
    assert Symmetric.braid(X, Y) == swap
    assert equal(f @ h >> Symmetric.swap(Y, W), Symmetric.swap(X, Z) >> h @ f)


def test_to_int():
    assert f.to_int().dom == IntTy[T](x, x_)
    assert f.to_int().cod == IntTy[T](y, y_)
    assert f.to_int().inside == f.forward @ y_ >> y @ f.backward
    assert (f >> g).to_int().inside.to_hypergraph()\
        == (f.to_int() >> g.to_int()).inside.to_hypergraph()
    assert Symmetric.id(X).to_int().inside.to_hypergraph()\
        == f.to_int().id(IntTy[T](x, x_)).inside.to_hypergraph()


def random_traced(seed):
    """
    Random lenses over a traced category: types of length up to two, the
    residual on either side of each leg and square pairs to trace over.
    """
    rng, names = random.Random(seed), iter(range(100))

    def ty(k=None):
        k = rng.randint(0, 2) if k is None else k
        return T(*rng.choices("abc", k=k))

    def pair(square=False):
        k = rng.randint(1, 2) if square else None
        return Ty(ty(k), ty(k))

    def lens_(dom, cod):
        (x, x_), (y, y_), m, name = dom, cod, ty(), f"f{next(names)}"
        forward = Box(name, x, m @ y) >> Diagram.swap(m, y)\
            if rng.random() < .5 else Box(name, x, y @ m)
        backward = Diagram.swap(m, y_) >> Box(name + "'", y_ @ m, x_)\
            if rng.random() < .5 else Box(name + "'", m @ y_, x_)
        return Traced(dom, cod, forward, backward, m)

    return pair, lens_


def natural_trace(lens_, U):
    """ The trace of `U` on the integer diagram in the underlying category. """
    (u, u_), swap = U, Diagram.swap
    x, y = (t.positive[:len(t.positive) - len(u)]
            for t in (lens_.dom, lens_.cod))
    x_, y_ = (t.negative[:len(t.negative) - len(u_)]
              for t in (lens_.dom, lens_.cod))
    inside = lens_.to_int().inside.trace(len(u_))
    return (x @ swap(y_, u) >> inside >> y @ swap(u, x_)).trace(len(u))


def test_traced():
    assert traced().trace(0) == traced()
    with raises(NotImplementedError):
        traced().trace(left=True)
    with raises(AxiomError):
        traced().trace()
    with raises(TypeError):
        traced() >> g


def test_trace_axioms():
    for seed in range(3):
        pair, lens_ = random_traced(seed)
        X, Y, X2, Y2, W = (pair() for _ in range(5))
        U, V = pair(square=True), pair(square=True)
        n, k = len(U.positive), len(V.positive)
        fu, gx, gy = lens_(X @ U, Y @ U), lens_(X2, X), lens_(Y, Y2)
        assert equal((gx @ Traced.id(U) >> fu).trace(n), gx >> fu.trace(n))
        assert equal((fu >> gy @ Traced.id(U)).trace(n), fu.trace(n) >> gy)
        assert equal(
            (Traced.id(W) @ fu).trace(n), Traced.id(W) @ fu.trace(n))
        assert equal(Traced.swap(U, U).trace(n), Traced.id(U))
        fuv = lens_(X @ U @ V, Y @ U @ V)
        assert equal(fuv.trace(k).trace(n), fuv.trace(n + k))
        fv, guv = lens_(X @ V, Y @ U), lens_(U, V)
        assert equal((fv >> Traced.id(Y) @ guv).trace(k),
                     (Traced.id(X) @ guv >> fv).trace(n))
        assert fu.trace(n).to_int().inside.to_hypergraph()\
            == natural_trace(fu, U).to_hypergraph()


def markov_lens():
    get = markov.Box('get', x, y)
    put = markov.Box('put', x @ y_, x_)
    return Markov(X, Y, get, put)


def test_markov_axioms():
    """
    The right unit law for `put` holds up to the naturality of discard,
    which the hypergraph does not decide, so it is checked on `get` only.
    """
    l = markov_lens()
    l_ = Markov(Y, Z, markov.Box('get_', y, z), markov.Box('put_', y @ z_, y_))
    assert (Markov.id(X) >> l).get == l.get
    assert (Markov.id(X) >> l).put.to_hypergraph() == l.put.to_hypergraph()
    assert (l >> Markov.id(Y)).get == l.get
    assert (l >> l_).get == l.get >> l_.get
    assert (l >> l_).put == markov.Diagram.copy(x) @ z_ >> x @ l.get @ z_\
        >> x @ l_.put >> l.put
    assert Markov.swap(X, Y).get == markov.Diagram.swap(x, y)
    assert Markov.swap(X, Y).put == markov.Diagram.discard(x @ y)\
        @ markov.Diagram.swap(y_, x_)
    assert Markov.braid(X, Y) == Markov.swap(X, Y)
    assert (l @ l_).dom == X @ Y and (l @ l_).cod == Y @ Z


def test_markov_associativity():
    """
    Composition is associative up to the naturality of copy for `get`,
    i.e. when `get` is deterministic as in a cartesian category: the left
    bracketing reads `x` twice with `get`, the right one reads it once and
    copies `y`. The hypergraph decides coassociativity but not naturality,
    so the right bracketing is compared with its copy pushed through `get`.
    """
    l, copy = markov_lens(), markov.Diagram.copy
    l_ = Markov(Y, Z, markov.Box('get_', y, z), markov.Box('put_', y @ z_, y_))
    l__ = Markov(
        Z, W, markov.Box('get__', z, w), markov.Box('put__', z @ w_, z_))
    left, right = (l >> l_) >> l__, l >> (l_ >> l__)
    assert left.get == right.get
    assert left.put.to_hypergraph() != right.put.to_hypergraph()
    natural = copy(x) @ w_ >> x @ copy(x) @ w_ >> x @ l.get @ l.get @ w_\
        >> x @ y @ l_.get @ w_ >> x @ y @ l__.put >> x @ l_.put >> l.put
    assert left.put.to_hypergraph() == natural.to_hypergraph()
    assert ((square >> square) >> square).put(2., 1.) == 1024.\
        == (square >> (square >> square)).put(2., 1.)


def test_markov_without_copy():
    get, put = Box('get', x, y), Box('put', x @ y_, x_)
    for method in (lambda: Markov[Diagram](X, Y, get, put),
                   lambda: Markov[Diagram].lift(get), f.to_markov):
        with raises(AxiomError, match="no copy or discard"):
            method()


def test_markov_is_symmetric():
    l = markov_lens()
    assert isinstance(l, Symmetric)
    assert l.residual == x and l.backward == l.put
    assert l.forward == markov.Diagram.copy(x) >> l.get @ x
    assert l.to_int().inside.to_hypergraph() == (
        markov.Diagram.copy(x) @ y_ >> l.get @ x @ y_ >> y @ l.put
    ).to_hypergraph()
    assert l.to_markov().get.to_hypergraph() == l.get.to_hypergraph()
    assert (l >> Markov.id(Y)).residual == x


R = Ty[python.Ty]((float, ), (float, ))
square = Markov[Function](
    R, R, Function(lambda a: a * a, (float, ), (float, )),
    Function(lambda a, da: 2 * a * da, (float, float), (float, )))


def test_chain_rule():
    assert (square >> square).get(3.) == 81.
    assert (square @ square).get(3., 3.) == (9., 9.)
    assert (square >> square).put(3., 1.) == 108.
    assert (square @ square).put(2., 3., 1., 1.) == (4., 6.)
    assert (square >> Markov[Function].id(R)).put(3., 1.) == 6.
    swapped = Markov[Function].swap(R, R) >> square @ square
    assert swapped.put(2., 3., 1., 1.) == (4., 6.)
    assert square.forward(3.) == (9., 3.)
    assert square.to_markov().put(3., 1.) == 6.


def test_learner():
    inside = Markov[Function](
        R @ R, R, Function(lambda a, w: w * a, (float, float), (float, )),
        Function(lambda a, w, da: (w * da, a * da),
                 (float, float, float), (float, float)))
    layer = para.Symmetric[Markov[Function]](R, R, inside, param=R)
    network = layer >> layer
    assert network.param == R @ R
    assert network.inside.get(2., 3., 5.) == 30.
    assert network.inside.put(2., 3., 5., 1.) == (15., 10., 6.)
    assert (layer @ layer).inside.get(1., 2., 3., 4.) == (3., 8.)
