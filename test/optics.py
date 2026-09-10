# -*- coding: utf-8 -*-

import random

from pytest import raises

from discopy import cat, markov, monoidal, optics, symmetric
from discopy.interaction import Ty as IntTy
from discopy.optics import Lens, Optic, Traced, Ty
from discopy.para import Symmetric
from discopy.python import Function
from discopy.symmetric import Box, Diagram
from discopy.symmetric import Ty as T
from discopy.utils import AxiomError

x, x_, y, y_, z, z_, w, w_ = map(
    T, ["x", "x'", "y", "y'", "z", "z'", "w", "w'"])
m, n, k = map(T, "mnk")
X, Y, Z, W = Ty(x, x_), Ty(y, y_), Ty(z, z_), Ty(w, w_)
f = Optic(X, Y, Box('f', x, y @ m), Box("f'", m @ y_, x_), m)
g = Optic(Y, Z, Box('g', y, z @ n), Box("g'", n @ z_, y_), n)
h = Optic(Z, W, Box('h', z, w @ k), Box("h'", k @ w_, z_), k)


def equal(left, right):
    """ Equality of optics up to sliding, decided by the hypergraph. """
    return (left.dom, left.cod) == (right.dom, right.cod)\
        and left.to_int().inside.to_hypergraph()\
        == right.to_int().inside.to_hypergraph()


def traced():
    return Traced(X, Y, f.forward, f.backward, m)


def test_ty():
    assert Ty[int](1, 2) @ Ty[int](3, 4) == Ty[int](4, 6)
    assert -(X @ Y) == -X @ -Y == Ty(x_ @ y_, x @ y)
    scope = {"cat": cat, "monoidal": monoidal, "optics": optics,
             "symmetric": symmetric, "markov": markov}
    assert eval(repr(X), scope) == X and str(X) == "x @ -x'"
    assert eval(repr(f), scope) == f and eval(repr(lens()), scope) == lens()
    assert eval(repr(traced()), scope) == traced() != f
    assert repr(square).startswith(
        "optics.Lens[python.multiplicative.Function](")
    assert repr(Optic[Function].id(R)).startswith(
        "optics.Optic[python.multiplicative.Function](")
    assert Ty[int]() == Ty[int](0, 0) and Ty.unit() == Ty()
    assert Ty[int].unit() == Ty[int]() and Ty[tuple].unit() == Ty[tuple]()
    assert Ty.negatives is tuple and IntTy.negatives is reversed


def test_errors():
    with raises(TypeError):
        Optic(x, y, Box('f', x, y), Box("f'", y_, x_))
    with raises(TypeError):
        Optic(IntTy[T](x, x_), Y, f.forward, f.backward, m)
    with raises(TypeError):
        Optic(X, Y, f.forward, f.backward, "m")
    with raises(TypeError):
        f >> Lens.id(Y)
    with raises(TypeError):
        f @ Lens.id(Y)
    with raises(TypeError):
        lens() >> Optic.id(Y)
    with raises(AxiomError):
        Optic(X, Y, Box('f', x, y @ m), Box("f'", m @ y_, x_), n)
    with raises(AxiomError):
        Optic(X, Z, Box('f', x, y @ m), Box("f'", m @ y_, x_), m)
    with raises(AxiomError):
        g >> f
    with raises(AxiomError):
        Lens(X, Y, markov.Box('get', x, y), markov.Box('put', x @ y_, x))


def test_category_axioms():
    assert f >> Optic.id(Y) == f == Optic.id(X) >> f
    assert equal((f >> g) >> h, f >> (g >> h))
    assert f >> g >> h == Optic.then(f, g, h)
    assert (f @ h).dom == X @ Z and (f @ h).cod == Y @ W
    assert (f @ h).residual == m @ k
    assert equal(f @ h.dom >> f.cod @ h, f @ h)
    assert equal((f @ g) >> (g @ h), (f >> g) @ (g >> h))
    assert equal((f @ h) @ g, f @ (h @ g))
    assert ((f @ h) @ g).residual == (f @ (h @ g)).residual
    assert f @ Optic.id() == f == Optic.id() @ f
    assert Optic.id() == Optic.lift(Diagram.id(T()))


def test_lift():
    forward, backward = Box('p', x, y), Box('q', y_, x_)
    assert Optic.lift(forward, backward) == Optic(X, Y, forward, backward)
    assert Optic.lift(forward) == Optic(Ty(x, T()), Ty(y, T()), forward,
                                        Diagram.id(T()))
    get, put = markov.Box('p', x, y), markov.Box('q', y_, x_)
    assert Lens.lift(get, put).put == markov.Diagram.discard(x) @ put


def test_symmetric_axioms():
    swap = Optic.swap(X, Y)
    assert swap.dom == X @ Y and swap.cod == Y @ X
    assert (swap >> Optic.swap(Y, X)).forward.simplify() == Diagram.id(x @ y)
    assert (swap >> Optic.swap(Y, X)).backward.simplify()\
        == Diagram.id(x_ @ y_)
    assert Optic.permutation((1, 0), (X, Y)).forward.simplify()\
        == swap.forward
    assert Optic.braid(X, Y) == swap
    assert equal(f @ h >> Optic.swap(Y, W), Optic.swap(X, Z) >> h @ f)


def test_to_int():
    assert f.to_int().dom == IntTy[T](x, x_)
    assert f.to_int().cod == IntTy[T](y, y_)
    assert f.to_int().inside == f.forward @ y_ >> y @ f.backward
    assert (f >> g).to_int().inside.to_hypergraph()\
        == (f.to_int() >> g.to_int()).inside.to_hypergraph()
    assert Optic.id(X).to_int().inside.to_hypergraph()\
        == f.to_int().id(IntTy[T](x, x_)).inside.to_hypergraph()


def random_traced(seed):
    """
    Random optics over a traced category: types of length up to two, the
    residual on either side of each leg and square pairs to trace over.
    """
    rng, names = random.Random(seed), iter(range(100))

    def ty(k=None):
        k = rng.randint(0, 2) if k is None else k
        return T(*rng.choices("abc", k=k))

    def pair(square=False):
        k = rng.randint(1, 2) if square else None
        return Ty(ty(k), ty(k))

    def optic(dom, cod):
        (x, x_), (y, y_), m, name = dom, cod, ty(), f"f{next(names)}"
        forward = Box(name, x, m @ y) >> Diagram.swap(m, y)\
            if rng.random() < .5 else Box(name, x, y @ m)
        backward = Diagram.swap(m, y_) >> Box(name + "'", y_ @ m, x_)\
            if rng.random() < .5 else Box(name + "'", m @ y_, x_)
        return Traced(dom, cod, forward, backward, m)

    return pair, optic


def natural_trace(optic, U):
    """ The trace of `U` on the integer diagram in the underlying category. """
    (u, u_), swap = U, Diagram.swap
    x, y = (t.positive[:len(t.positive) - len(u)]
            for t in (optic.dom, optic.cod))
    x_, y_ = (t.negative[:len(t.negative) - len(u_)]
              for t in (optic.dom, optic.cod))
    inside = optic.to_int().inside.trace(len(u_))
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
        pair, optic = random_traced(seed)
        X, Y, X2, Y2, W = (pair() for _ in range(5))
        U, V = pair(square=True), pair(square=True)
        n, k = len(U.positive), len(V.positive)
        fu, gx, gy = optic(X @ U, Y @ U), optic(X2, X), optic(Y, Y2)
        assert equal((gx @ Traced.id(U) >> fu).trace(n), gx >> fu.trace(n))
        assert equal((fu >> gy @ Traced.id(U)).trace(n), fu.trace(n) >> gy)
        assert equal(
            (Traced.id(W) @ fu).trace(n), Traced.id(W) @ fu.trace(n))
        assert equal(Traced.swap(U, U).trace(n), Traced.id(U))
        fuv = optic(X @ U @ V, Y @ U @ V)
        assert equal(fuv.trace(k).trace(n), fuv.trace(n + k))
        fv, guv = optic(X @ V, Y @ U), optic(U, V)
        assert equal((fv >> Traced.id(Y) @ guv).trace(k),
                     (Traced.id(X) @ guv >> fv).trace(n))
        assert fu.trace(n).to_int().inside.to_hypergraph()\
            == natural_trace(fu, U).to_hypergraph()


def lens():
    get = markov.Box('get', x, y)
    put = markov.Box('put', x @ y_, x_)
    return Lens(X, Y, get, put)


def test_lens_axioms():
    """
    The right unit law for `put` holds up to the naturality of discard,
    which the hypergraph does not decide, so it is checked on `get` only.
    """
    l = lens()
    l_ = Lens(Y, Z, markov.Box('get_', y, z), markov.Box('put_', y @ z_, y_))
    assert (Lens.id(X) >> l).get == l.get
    assert (Lens.id(X) >> l).put.to_hypergraph() == l.put.to_hypergraph()
    assert (l >> Lens.id(Y)).get == l.get
    assert (l >> l_).get == l.get >> l_.get
    assert (l >> l_).put == markov.Diagram.copy(x) @ z_ >> x @ l.get @ z_\
        >> x @ l_.put >> l.put
    assert Lens.swap(X, Y).get == markov.Diagram.swap(x, y)
    assert Lens.swap(X, Y).put == markov.Diagram.discard(x @ y)\
        @ markov.Diagram.swap(y_, x_)
    assert Lens.braid(X, Y) == Lens.swap(X, Y)
    assert (l @ l_).dom == X @ Y and (l @ l_).cod == Y @ Z


def test_lens_associativity():
    """
    Composition is associative up to the naturality of copy for `get`,
    i.e. when `get` is deterministic as in a cartesian category: the left
    bracketing reads `x` twice with `get`, the right one reads it once and
    copies `y`. The hypergraph decides coassociativity but not naturality,
    so the right bracketing is compared with its copy pushed through `get`.
    """
    l, copy = lens(), markov.Diagram.copy
    l_ = Lens(Y, Z, markov.Box('get_', y, z), markov.Box('put_', y @ z_, y_))
    l__ = Lens(
        Z, W, markov.Box('get__', z, w), markov.Box('put__', z @ w_, z_))
    left, right = (l >> l_) >> l__, l >> (l_ >> l__)
    assert left.get == right.get
    assert left.put.to_hypergraph() != right.put.to_hypergraph()
    natural = copy(x) @ w_ >> x @ copy(x) @ w_ >> x @ l.get @ l.get @ w_\
        >> x @ y @ l_.get @ w_ >> x @ y @ l__.put >> x @ l_.put >> l.put
    assert left.put.to_hypergraph() == natural.to_hypergraph()
    assert ((square >> square) >> square).put(2., 1.) == 1024.\
        == (square >> (square >> square)).put(2., 1.)


def test_lens_not_markov():
    get, put = Box('get', x, y), Box('put', x @ y_, x_)
    l = Lens[Diagram](X, Y, get, put)
    l_ = Lens[Diagram](Y, Z, Box('get_', y, z), Box('put_', y @ z_, y_))
    assert l.get == get and l.put == put
    for method in (lambda: l >> l_, lambda: Lens[Diagram].id(X),
                   lambda: Lens[Diagram].lift(get), l.to_optic, f.to_lens):
        with raises(AxiomError, match="no copy or discard"):
            method()
    discard = lambda typ: Box('discard', typ, T())
    assert f.to_lens(discard).get == f.forward >> y @ discard(m)
    assert f.to_lens(discard).put\
        == f.forward @ y_ >> discard(y) @ m @ y_ >> f.backward
    assert f.to_lens(discard) == Lens[Diagram](
        X, Y, f.to_lens(discard).get, f.to_lens(discard).put)


def test_lens_and_optic():
    l = lens()
    optic = l.to_optic()
    assert optic.residual == x and optic.backward == l.put
    assert optic.to_int().inside.to_hypergraph() == (
        markov.Diagram.copy(x) @ y_ >> l.get @ x @ y_ >> y @ l.put
    ).to_hypergraph()
    assert optic.to_lens().get.to_hypergraph() == l.get.to_hypergraph()


R = Ty[tuple]((float, ), (float, ))
square = Lens[Function](
    R, R, Function(lambda a: a * a, (float, ), (float, )),
    Function(lambda a, da: 2 * a * da, (float, float), (float, )))


def test_chain_rule():
    assert (square >> square).get(3.) == 81.
    assert (square @ square).get(3., 3.) == (9., 9.)
    assert (square >> square).put(3., 1.) == 108.
    assert (square @ square).put(2., 3., 1., 1.) == (4., 6.)
    assert (square >> Lens[Function].id(R)).put(3., 1.) == 6.
    assert (Lens[Function].swap(R, R) >> square @ square).put(2., 3., 1., 1.)\
        == (4., 6.)
    optic = square.to_optic()
    assert optic.forward(3.) == (9., 3.) and optic.to_lens().put(3., 1.) == 6.


def test_learner():
    inside = Lens[Function](
        R @ R, R, Function(lambda a, w: w * a, (float, float), (float, )),
        Function(lambda a, w, da: (w * da, a * da),
                 (float, float, float), (float, float)))
    layer = Symmetric[Lens[Function]](R, R, inside, param=R)
    network = layer >> layer
    assert network.param == R @ R
    assert network.inside.get(2., 3., 5.) == 30.
    assert network.inside.put(2., 3., 5., 1.) == (15., 10., 6.)
    assert (layer @ layer).inside.get(1., 2., 3., 4.) == (3., 8.)
