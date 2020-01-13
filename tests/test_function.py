# -*- coding: utf-8 -*-

from pytest import raises
from discopy.function import *


def test_Vec_init():
    with raises(ValueError) as err:
        Vec(-1)
    assert "Expected non-negative integer, got -1" in str(err.value)


def test_Vec_repr():
    assert repr(Vec(5)) == "Vec(5)"


def test_Vec_str():
    assert str(Vec(0) @ Vec(3)) == "Vec(3)"


def test_Vec_hash():
    dim = Vec(3)
    assert {dim: 42}[dim] == 42


def test_Function_init():
    f = Function('f', Vec(2), Vec(2), lambda x: x)
    with raises(TypeError) as err:
        Function('f', 2., Vec(3), lambda x: x)
    assert "Expected discopy.function.Vec, got 2.0" in str(err.value)
    with raises(TypeError) as err:
        Function('f', Vec(2), Ob('x'), lambda x: x)
    assert "Expected discopy.function.Vec, got Ob('x')" in str(err.value)
    with raises(TypeError) as err:
        Function(5, Vec(2), Vec(1), lambda x: x)
    assert "Expected builtins.str, got 5 of type int instead" in str(err.value)


def test_Function_repr():
    id = Function('Id_2', Vec(2), Vec(2), lambda x: x)
    assert 'Function(name=Id_2, dom=Vec(2), cod=Vec(2)' in repr(id)


def test_Function_str():
    assert str(Function('copy', Vec(2), Vec(2), lambda x: x + x)) == "copy"


def test_Function_call():
    with raises(AxiomError) as err:
        Function('f', Vec(2), Vec(2), lambda x: x)([3])
    assert "Expected input of length 2, got 1 instead." in str(err.value)


def test_Function_then():
    id = Function('id', Vec(3), Vec(3), lambda x: x)
    swap = Function('swap', Vec(2), Vec(2), lambda x: x[::-1])
    with raises(AxiomError) as err:
        swap >> id
    assert "does not compose" in str(err.value)


def test_Function_tensor():
    with raises(TypeError) as err:
        Function('id', Vec(3), Vec(3), lambda x: x) >> (lambda x: x)
    assert "Expected discopy.function.Function, got <functio" in str(err.value)


def test_AxiomError():
    with raises(AxiomError) as err:
        Function('f', Vec(0), Vec(2), lambda x: x)([3, 2])
    assert "Expected input of length 0, got 2 instead." in str(err.value)
