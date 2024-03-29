from discopy import *
from discopy.feedback import *


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

def test_Functor():
    from discopy import stream, python

x = Ty('int')
zero, one = Box('0', Ty(), x.head), Box('1', Ty(), x.head)
plus = Box('+', x @ x, x)
fib = ((Copy(x) >> one @ Diagram.wait(x) @ x
        >> FollowedBy(x) @ x >> plus).delay()
        >> zero @ x.delay() >> FollowedBy(x) >> Copy(x)).feedback()

F = Functor(
    ob={x: int},
    ar={zero: lambda: 0, one: lambda: 1,
        plus: lambda x, y: x + y},
    cod=stream.Category(python.Ty, python.Function))
assert F(fib).unroll(5).now() == (0, 1, 1, 2, 3)
