from discopy import cartesian, cat, markov, monoidal  # noqa: F401
from discopy.abc import CartesianCategory
from discopy.cartesian import *


def test_hierarchy():
    assert issubclass(Diagram, markov.Diagram)
    assert issubclass(Diagram, CartesianCategory)
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
