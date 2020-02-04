from pytest import raises
from discopy.cartesian import *


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
    f = Function(2, 1, lambda x, y: x + y)
    g = (lambda x: x, )
    with raises(TypeError) as err:
        f @ g
    assert str(err.value) == messages.type_err(Function, g)
