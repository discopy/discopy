from pytest import raises
from discopy.cartesian import *


def test_Box_repr():
    f = Box('f', 1, 2, lambda x: (x, x))
    assert "Box('f', 1, 2" in repr(f)


def test_Function_str():
    f = Function(2, 1, lambda x, y: x + y)
    assert 'Function(dom=2, cod=1,' in str(f)


def test_Function_call():
    f = Swap(2, 1)
    values = (2, 3)
    with raises(TypeError) as err:
        f(*values)
    assert str(err.value) == messages.expected_input_length(f, values)


def test_Function_then():
    f, g = Function(2, 1, lambda x, y: x + y), Function(1, 1, lambda x: x + 1)
    assert Function.id(2).then(*(f, g))(20, 21) == 42


def test_Function_then_err():
    f = Function(2, 1, lambda x, y: x + y)
    g = (lambda x: x, )
    with raises(TypeError) as err:
        f >> g
    assert str(err.value) == messages.type_err(Function, g)
    g = Function.id(2)
    with raises(AxiomError) as err:
        f >> g
    assert str(err.value) == messages.does_not_compose(f, g)


def test_Function_tensor():
    assert Function.id(3)(1, 2, 3)\
        == Function.id(0).tensor(*(3 * [Function.id(1)]))(1, 2, 3)


def test_Function_tensor_err():
    f = Function(2, 1, lambda x, y: x + y)
    g = (lambda x: x, )
    with raises(TypeError) as err:
        f @ g
    assert str(err.value) == messages.type_err(Function, g)
