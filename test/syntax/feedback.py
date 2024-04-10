from pytest import raises

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


def test_fibonacci():
    x = Ty('int')
    zero, one, plus = Box('0', Ty(), x), Box('1', Ty(), x), Box('+', x @ x, x)
    fib = ((Copy(x) >> one.head @ Diagram.wait(x) @ x
            >> FollowedBy(x) @ x >> plus).delay()
            >> zero.head @ x.delay() >> FollowedBy(x) >> Copy(x)).feedback()

    F = Functor(
        ob={x: int},
        ar={zero: lambda: 0,
            one: lambda: 1,
            plus: lambda x, y: x + y},
        cod=stream.Category(python.Ty, python.Function))

    F(Diagram.wait(x))
    assert F(zero >> Diagram.wait(x)).unroll(5).now() == ((), 0, 0, 0, 0)
    assert F(fib).unroll(5).now() == (0, 1, 1, 2, 3)
