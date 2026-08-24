# -*- coding: utf-8 -*-
"""B26: Substitution returns None on constants and recurses forever on abstractions (discopy/closed.py:285).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.closed import Abstraction, Constant, Substitution, Ty, Variable


def test_b26_substitute_constant():
    x = Ty('x')
    c, v = Constant('c', x), Variable('v', x)
    assert Substitution({v: c})(c) == c


def test_b26_substitute_abstraction():
    x = Ty('x')
    v, w = Variable('v', x), Variable('w', x)
    term = Abstraction(v, v)
    result = Substitution({w: Constant('c', x)})(term)
    assert isinstance(result, Abstraction)
