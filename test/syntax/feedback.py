from pytest import raises
from random import choice, seed

from discopy import *
from discopy.feedback import *


def test_invalid_inputs():
    with raises(NotImplementedError):
        Ty('x').delay(-1)


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


X = Ty('X')
fby, wait = FollowedBy(X), Swap(X, X.d).feedback()
zero, one = Box('0', Ty(), X), Box('1', Ty(), X)
copy, plus = Copy(X), Box('+', X @ X, X)


@Diagram.feedback
@Diagram.from_callable(X.d, X @ X)
def fib(x):
    y = fby(zero.head(), plus.d(fby.d(one.head.d(), wait.d(x)), x))
    return (y, y)


def test_fibonacci_eq():
    with Diagram.hypergraph_equality:
        assert fib == (
            copy.d >> one.head.d @ wait.d @ X.d
                >> fby.d @ X.d
                >> plus.d
                >> zero.head @ X.d
                >> fby >> copy).feedback()
