# -*- coding: utf-8 -*-

from pytest import raises
from discopy.function import *


def test_Function_init():
    f = Function(PRO(2), PRO(2), lambda x: x)
    with raises(TypeError) as err:
        Function(2., PRO(3), lambda x: x)
    assert "Expected builtins.int, got 2.0" in str(err.value)
    with raises(TypeError) as err:
        Function(PRO(1), 4., lambda x: x)
    assert "Expected builtins.int, got 4.0" in str(err.value)


def test_Function_repr():
    id = Function(PRO(2), PRO(2), lambda x: x)
    assert 'Function(dom=PRO(2), cod=PRO(2)' in repr(id)


def test_Function_str():
    f = Function(PRO(2), PRO(2), lambda x: x + x)
    assert str(f) == repr(f)


def test_Function_call():
    with raises(AxiomError) as err:
        Function(PRO(2), PRO(2), lambda x: x)([3])
    assert "Expected input of length 2, got 1 instead." in str(err.value)


def test_Function_then():
    id = Function(PRO(3), PRO(3), lambda x: x)
    swap = Function(PRO(2), PRO(2), lambda x: x[::-1])
    with raises(AxiomError) as err:
        swap >> id
    assert "does not compose" in str(err.value)
    with raises(TypeError) as err:
        Function(PRO(3), PRO(3), lambda x: x) >> (lambda x: x)
    assert "Expected discopy.function.Function, got <functio" in str(err.value)


def test_Function_tensor():
    with raises(TypeError) as err:
        Function(PRO(3), PRO(3), lambda x: x) @ (lambda x: x)
    assert "Expected discopy.function.Function, got <functio" in str(err.value)


def test_AxiomError():
    with raises(AxiomError) as err:
        Function(PRO(0), PRO(2), lambda x: x)([3, 2])
    assert "Expected input of length 0, got 2 instead." in str(err.value)
