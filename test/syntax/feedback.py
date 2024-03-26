from discopy import *
from discopy.feedback import *


def test_Diagram_repr():
    x = Ty('x')
    plus = Box('plus', x @ x, x)
    zero = Box('zero', Ty(), x.head)
    zero, one = Box('zero', Ty(), x.head), Box('one', Ty(), x.head)
    fib = (
        Copy(x) >> (one @ Diagram.wait(x)).delay() @ x
        >> FollowedBy(x).delay() @ x >> zero @ plus.delay()
        >> FollowedBy(x) >> Copy(x)).feedback()
    assert eval(str(fib)) == fib
    assert eval(repr(fib)) == fib
