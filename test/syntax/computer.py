from __future__ import annotations
from collections.abc import Callable

from discopy.computer import *
from discopy import *


def test_python_Functor():
    x = Ty('x')
    copy, discard, eval = Copy(x), Discard(x), Eval(x, x)
    add, minus = Box('+', x @ x, x), Box('-', x, x)

    from discopy.python import Function
    F = Functor(
        ob={x: int, P: Callable[[int], tuple[int]]},
        ar={add: lambda x, y: x+y,
            minus: lambda x: x-1,},
        cod=Category(tuple[type, ...], Function))

    f = eval >> copy >> minus @ x >> add >> copy >> (minus @ discard)

    assert F(f)(lambda x: x**3, 2) == 14
