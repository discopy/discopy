from pytest import raises
from discopy.matrix import *


def test_AxiomError():
    m = Matrix(Dim(2, 2), Dim(2), [1, 0, 0, 1, 0, 1, 1, 0])
    with raises(AxiomError) as err:
        m >> m
    assert str(err.value) == config.Msg.does_not_compose(m, m)
