# -*- coding: utf-8 -*-

from pytest import raises
from discopy.function import *


def test_Dim_init():
    with raises(ValueError) as err:
        Dim(-1)
    assert "Expected non-negative integer, got -1" in str(err.value)


def test_Dim_repr():
    assert repr(Dim(5)) == "Dim(5)"


def test_Dim_str():
    assert str(Dim(0) @ Dim(3)) == "Dim(3)"


def test_Dim_hash():
    dim = Dim(3)
    assert {dim: 42}[dim] == 42


def test_Function_init():
    f = Function('f', Dim(2), Dim(2), lambda x: x)
    with raises(ValueError) as err:
        Function('f', 2., Dim(3), lambda x: x)
    assert "Dim expected for dom, got <class 'float'>" in str(err.value)
    with raises(ValueError) as err:
        Function('f', Dim(2), Ob('x'), lambda x: x)
    assert "Dim expected for cod, got <class 'discopy.cat.Ob" in str(err.value)
    with raises(ValueError) as err:
        Function(5, Dim(2), Dim(1), lambda x: x)
    assert "String expected for name, got <class 'int'>" in str(err.value)


def test_Function_repr():
    id = Function('Id_2', Dim(2), Dim(2), lambda x: x)
    assert 'Function(name=Id_2, dom=Dim(2), cod=Dim(2)' in repr(id)


def test_Function_str():
    assert str(Function('copy', Dim(2), Dim(2), lambda x: x + x)) == "copy"


def test_Function_call():
    with raises(AxiomError) as err:
        Function('f', Dim(2), Dim(2), lambda x: x)([3])
    assert "Expected input of length 2, got 1 instead." in str(err.value)


def test_Function_then():
    id = Function('id', Dim(3), Dim(3), lambda x: x)
    swap = Function('swap', Dim(2), Dim(2), lambda x: x[::-1])
    with raises(AxiomError) as err:
        swap >> id
    assert "does not compose" in str(err.value)


def test_Function_tensor():
    with raises(ValueError) as err:
        Function('id', Dim(3), Dim(3), lambda x: x) >> (lambda x: x)
    assert "Function expected, got <function" in str(err.value)


def test_AxiomError():
    with raises(AxiomError) as err:
        Function('f', Dim(0), Dim(2), lambda x: x)([3, 2])
    assert "Expected input of length 0, got 2 instead." in str(err.value)
