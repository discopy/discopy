from __future__ import annotations
from pytest import raises

from discopy.python import Function

from discopy.markov import *
from discopy import *


def test_spider_factory():
    with raises(ValueError):
        Diagram.spider_factory(2, 2, Ty('x'))


def test_Merge_dagger():
    assert Merge(Ty('x')).dagger() == Copy(Ty('x'))


def test_Discard():
    assert isinstance(Discard(Ty('x')), Discard)
    assert isinstance(Copy(Ty('x'), n=0), Discard)


def test_equations():
    x = Ty('x')
    copy, discard = Copy(x), Copy(x, 0)
    add, minus, zero = Box('+', x @ x, x), Box('-', x, x), Box('0', Ty(), x)

    add >> copy, copy @ copy >> x @ Swap(x, x) @ x >> add @ add
    add >> discard, discard @ discard
    zero >> discard, Diagram.id(Ty())
    copy >> minus @ x >> add, discard >> zero, copy >> x @ minus >> add

    Diagram.id(x)
    x @ zero >> x @ copy >> add @ x >> discard @ x
    x @ zero @ zero >> discard @ discard @ x
    discard >> zero


def test_neural_network():
    x = Ty('x')
    add = lambda n: Box('$+$', x ** n, x)
    ReLU = Box('$\\sigma$', x, x)
    weights = [Box('w{}'.format(i), x, x) for i in range(4)]
    bias = Box('b', Ty(), x)

    network = Diagram.copy(x @ x, 2)\
    >> Diagram.tensor(*weights) @ bias >> add(5) >> ReLU

    F = Functor(ob_map={x: int}, ar_map={
            add(5): lambda *xs: sum(xs),
            ReLU: lambda x: max(0, x),
            bias: lambda: -1, **{
                weight: lambda x, w=w: x * w
                for weight, w in zip(weights, range(4))}},
        cod=Function)

    assert F(network)(42, 43) == max(0, sum([42 * 0, 43 * 1, 42 * 2, 43 * 3, -1]))


def test_Permutation():
    x, y, z = map(Ty, "xyz")
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert isinstance(perm, Box) and perm.cod == z @ x @ y
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    assert isinstance(perm.inside[0], Layer)
    assert Box('f', x, y).inside[0].boxes_or_types == (Box('f', x, y), )
    assert type(perm.inside[0].boxes_and_types[1]) is Permutation
    assert Permutation(x @ y, [1, 0]) == Swap(x, y)
    assert issubclass(Swap, Permutation)
    assert Equation(perm, perm.to_swaps())


def test_term_copy_and_discard():
    """ A shared variable is copied, an unused one is discarded — the
    non-linear behaviour that used to live in `discopy.closed`. """
    X, Y = Ty('X'), Ty('Y')
    x, y = Variable('x', X), Variable('y', Y)
    g = Constant('g', X @ X, Y)
    copied = g(x, x).eval()
    assert any(isinstance(box, Copy) for box in copied.boxes)
    discarded = g(x, x).eval(context=Context([x, y]))
    assert any(isinstance(box, Discard) for box in discarded.boxes)
    assert discarded.dom == X @ Y and discarded.cod == Y


def test_term_freevars_order():
    """ Free variables keep first-occurrence order rather than going
    through a set, whose iteration order depends on hashing, see #543. """
    A, B, C, W, Z = map(Ty, "ABCWZ")
    a, b, c = Variable('a', A), Variable('b', B), Variable('c', C)
    f, F = Constant('f', A @ B @ C, W), Constant('F', W @ A, Z)
    term = F(f(a, b, c), a)
    assert [x.name for x in term.freevars] == ['a', 'b', 'c']
    assert term.dom == A @ B @ C and term.cod == Z


def test_term_functor():
    """ A term with a shared variable evaluates to a python function that
    copies its argument, the migration of #562's example. """
    from discopy.python import Function
    X, Y = Ty('X'), Ty('Y')
    x, g = Variable('x', X), Constant('g', X @ X, Y)
    F = Functor(
        ob_map={X: int, Y: str},
        ar_map={g: lambda n, m: f"{n}|{m}"}, cod=Function)
    assert F(g(x, x).eval())(7) == "7|7"


def test_context_dom():
    """
    `Context.dom` instantiates `category.ob` before calling `.tensor`, so
    it works both for an empty context (regression test for #549) and for
    a non-empty one.
    """
    X = Ty('X')
    assert Context([]).dom == Ty()
    assert Context([Variable('x', X)]).dom == X
