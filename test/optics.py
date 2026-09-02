# -*- coding: utf-8 -*-

from pytest import raises

from discopy import cat, markov, monoidal, optics
from discopy.interaction import Ty as IntTy
from discopy.optics import Lens, Optic, Ty
from discopy.para import Symmetric
from discopy.python import Function
from discopy.symmetric import Box, Diagram
from discopy.symmetric import Ty as T
from discopy.utils import AxiomError

x, x_, y, y_, z, z_, w, w_ = map(
    T, ["x", "x'", "y", "y'", "z", "z'", "w", "w'"])
m, n, k = map(T, "mnk")
X, Y, Z, W = Ty(x, x_), Ty(y, y_), Ty(z, z_), Ty(w, w_)
f = Optic(X, Y, Box('f', x, m @ y), Box("f'", m @ y_, x_), m)
g = Optic(Y, Z, Box('g', y, n @ z), Box("g'", n @ z_, y_), n)
h = Optic(Z, W, Box('h', z, k @ w), Box("h'", k @ w_, z_), k)


def test_ty():
    assert Ty[int](1, 2) @ Ty[int](3, 4) == Ty[int](4, 6)
    assert -(X @ Y) == -X @ -Y == Ty(x_ @ y_, x @ y)
    scope = {"cat": cat, "monoidal": monoidal, "optics": optics}
    assert eval(repr(X), scope) == X and str(X) == "x @ -x'"
    assert Ty[int]() == Ty[int](0, 0) and Ty.unit() == Ty()


def test_errors():
    with raises(TypeError):
        Optic(x, y, Box('f', x, y), Box("f'", y_, x_))
    with raises(AxiomError):
        Optic(X, Y, Box('f', x, m @ y), Box("f'", m @ y_, x_), n)
    with raises(AxiomError):
        Optic(X, Z, Box('f', x, m @ y), Box("f'", m @ y_, x_), m)
    with raises(AxiomError):
        g >> f
    with raises(AxiomError):
        Lens(X, Y, markov.Box('get', x, y), markov.Box('put', x @ y_, x))


def test_category_axioms():
    assert f >> Optic.id(Y) == f == Optic.id(X) >> f
    assert (f >> g) >> h == f >> (g >> h)
    assert f >> g >> h == Optic.then(f, g, h)
    assert (f @ h).dom == X @ Z and (f @ h).cod == Y @ W
    assert (f @ h).residual == m @ k
    assert (f @ h.dom >> f.cod @ h).to_int().inside.to_hypergraph()\
        == (f @ h).to_int().inside.to_hypergraph()
    assert Optic.id() == Optic.lift(Diagram.id(T()))


def test_symmetric_axioms():
    swap = Optic.swap(X, Y)
    assert swap.dom == X @ Y and swap.cod == Y @ X
    assert (swap >> Optic.swap(Y, X)).forward.simplify() == Diagram.id(x @ y)
    assert (swap >> Optic.swap(Y, X)).backward.simplify()\
        == Diagram.id(x_ @ y_)
    assert Optic.permutation((1, 0), (X, Y)).forward.simplify()\
        == swap.forward
    assert Optic.braid(X, Y) == swap
    left, right = f @ h >> Optic.swap(Y, W), Optic.swap(X, Z) >> h @ f
    assert left.to_int().inside.to_hypergraph()\
        == right.to_int().inside.to_hypergraph()


def test_to_int():
    assert f.to_int().dom == IntTy[T](x, x_)
    assert f.to_int().cod == IntTy[T](y, y_)
    assert (f >> g).to_int().inside.to_hypergraph()\
        == (f.to_int() >> g.to_int()).inside.to_hypergraph()
    assert Optic.id(X).to_int().inside.to_hypergraph()\
        == f.to_int().id(IntTy[T](x, x_)).inside.to_hypergraph()


def lens():
    get = markov.Box('get', x, y)
    put = markov.Box('put', x @ y_, x_)
    return Lens(X, Y, get, put)


def test_lens_axioms():
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


def test_lens_and_optic():
    l = lens()
    optic = l.to_optic()
    assert optic.residual == x and optic.backward == l.put
    assert optic.to_int().inside.to_hypergraph() == (
        markov.Diagram.copy(x) @ y_ >> x @ l.get @ y_
        >> markov.Diagram.swap(x, y) @ y_ >> y @ l.put).to_hypergraph()
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
    assert optic.forward(3.) == (3., 9.) and optic.to_lens().put(3., 1.) == 6.


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
