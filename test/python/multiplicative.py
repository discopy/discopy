# -*- coding: utf-8 -*-

from typing import List
from pytest import raises

from discopy.biclosed import *
from discopy.python import *


def test_fixed_point():
    from math import sqrt
    phi = Function(lambda x=1: 1 + 1 / x, dom=(float,), cod=(float,)).fix()
    assert phi() == (1 + sqrt(5)) / 2


def test_trace():
    with raises(NotImplementedError):
        Function.id(int).trace(left=True)


def test_list_generic_in_function():
    func = Function(sum, List[int], int)
    assert func([1, 2, 3]) == 6


def test_strategy():
    from hypothesis import find
    from discopy.python.multiplicative import Function

    find(Function.strategy(), lambda f: len(f.dom) >= 2 and len(f.cod) >= 2)


def test_axioms():
    from discopy import testing
    from discopy.python.multiplicative import Function

    testing.assert_axioms(Function)
