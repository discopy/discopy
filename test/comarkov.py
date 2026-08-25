from __future__ import annotations
from pytest import raises

from discopy.comarkov import *
from discopy.utils import AxiomError


def test_spider_factory():
    with raises(ValueError):
        Diagram.spider_factory(2, 2, Ty('x'))
    with raises(ValueError):
        Diagram.spider_factory(1, 2, Ty('x'))


def test_Merge_dagger():
    with raises(AxiomError):
        Merge(Ty('x')).dagger()


def test_Unit():
    assert isinstance(Unit(Ty('x')), Unit)
    assert isinstance(Merge(Ty('x'), n=0), Unit)


def test_unit():
    x = Ty('x')
    assert Equation(Diagram.unit(x @ x), Merge(x, 0) @ Merge(x, 0))


def test_repr():
    assert repr(Merge(Ty('x'))) == \
        "comarkov.Merge(monoidal.Ty(cat.Ob('x')), 2)"


def test_functor():
    x = Ty('x')
    G = Functor(lambda ob: ob, lambda box: box)
    assert Equation(G(Merge(x)), Merge(x))
