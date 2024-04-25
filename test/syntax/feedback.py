from pytest import raises
from random import choice, seed

from discopy import *
from discopy.feedback import *


def test_invalid_inputs():
    with raises(NotImplementedError):
        Ty('x').delay(-1)
    with raises(ValueError):
        HeadOb(Ob('x').delay())
    with raises(ValueError):
        TailOb(Ob('x').delay())


def test_Diagram_repr():
    x = Ty('x')
    plus = Box('plus', x @ x, x)
    zero = Box('zero', Ty(), x.head)
    zero, one = Box('zero', Ty(), x.head), Box('one', Ty(), x.head)
    fib =  ((
            Copy(x) >> one @ Diagram.wait(x) @ x
            >> FollowedBy(x) @ x >> plus).delay()
        >> zero @ x.delay() >> FollowedBy(x) >> Copy(x)).feedback()
    assert eval(str(fib)) == fib
    assert eval(repr(fib)) == fib


def test_functor_python_stream():
    x = Ty('x')
    zero, wait = Box('zero', Ty(), x), Diagram.wait(x)
    F = Functor(
        ob={x: int},
        ar={zero: lambda: 0},
        cod=stream.Category(python.Ty, python.Function))
    assert F(wait @ zero).unroll(2).now(1, 2, 3) == (0, ) + (1, 0) + (2, 0) + (3, )


def test_walk():
    seed(420)
    X, fby = Ty('X'), FollowedBy(Ty('X'))
    zero, rand, plus = Box('0', Ty(), X), Box('rand', Ty(), X), Box('+', X @ X, X)

    @Diagram.feedback
    @Diagram.from_callable(X.d, X @ X)
    def walk(x):
        y = fby(zero.head(), plus.d(rand.d(), x))
        return (y, y)

    F = Functor(
        ob={X: int},
        ar={zero: lambda: 0,
            rand: lambda: choice([-1, +1]),
            plus: lambda x, y: x + y},
        cod=stream.Category(python.Ty, python.Function))

    assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 2, 1, 0, -1, 0, 1)
    assert F(walk).unroll(9).now()[:10] == (0, -1, -2, -1, 0, 1, 0, 1, 2, 1)
    assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 0, 1, 0, -1, 0, -1)

def test_fibonacci():
    X = Ty('X')
    fby, wait = FollowedBy(X), Swap(X, X.d).feedback()
    zero, one = Box('0', Ty(), X), Box('1', Ty(), X)
    copy, plus = Copy(X), Box('+', X @ X, X)


    @Diagram.feedback
    @Diagram.from_callable(X.d, X @ X)
    def fib(x):
        y = fby(zero.head(), plus.d(fby.d(one.head.d(), wait.d(x)), x))
        return (y, y)

    with Diagram.hypergraph_equality:
        assert fib == (
            copy.d >> one.head.d @ wait.d @ X.d
                >> fby.d @ X.d
                >> plus.d
                >> zero.head @ X.d
                >> fby >> copy).feedback()

    F = Functor(
        ob={X: (int, )},
        ar={zero: lambda: 0,
            one: lambda: 1,
            plus: lambda x, y: x + y},
        cod=stream.Category(python.Ty, python.Function))

    assert F(fib).unroll(9).now()[:10] == (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)
