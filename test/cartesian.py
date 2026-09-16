from discopy import cartesian, cat, closed, markov, monoidal  # noqa: F401
from discopy.abc import CartesianCategory, CartesianClosedCategory
from discopy.cartesian import *


def test_hierarchy():
    assert issubclass(Diagram, markov.Diagram)
    assert issubclass(Diagram, closed.Diagram)
    assert issubclass(Diagram, CartesianClosedCategory)
    assert not issubclass(markov.Diagram, CartesianCategory)


def test_factories():
    x = Ty('x')
    assert isinstance(Diagram.copy(x), Diagram)
    assert isinstance(Copy(x).dagger(), Merge)
    assert Copy(x, 0) == Discard(x)
    assert isinstance(Diagram.swap(x, x), Diagram)
    assert Copy(x).ar is Diagram


def test_terms():
    X, Y = Ty('X'), Ty('Y')
    x, f = Variable('x', X), Constant('f', X @ X, Y)
    term = f(x, x)
    assert isinstance(term, Application)
    assert isinstance(term.eval(), Diagram)
    assert term.eval() == Copy(X) >> f
    assert eval(repr(term)) == term


def test_projection():
    x, y, z = map(Ty, "xyz")
    p = Diagram.projection(x @ y @ z, 1)
    assert isinstance(p, Projection)
    assert (p.dom, p.cod) == (x @ y @ z, y)
    assert eval(repr(p)) == p
    assert Functor.id(Diagram)(p) == p
    from pytest import raises
    with raises(IndexError):
        Projection(x @ y, 2)

    from discopy import python
    F = Functor({x: int, y: bool, z: str}, {}, cod=python.Function)
    assert F(p)(42, True, "!") is True

    from discopy import cartesian_feedback
    X = cartesian_feedback.Ty('X')
    q = cartesian_feedback.Diagram.projection(X @ X, 0)
    assert (q.dom, q.cod) == (X @ X, X)
    assert all(isinstance(box, cartesian_feedback.Discard) for box in q.boxes)
