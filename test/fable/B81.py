# -*- coding: utf-8 -*-
"""B81: categorial FC/BC/FX/BX are not boxes, nested cfg.Tree.__call__ crashes, to_compact needs a dagger (discopy/grammar/categorial.py:318, cfg.py:99, closed.py:105).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy import closed
from discopy.grammar import cfg
from discopy.grammar.categorial import (
    BC, BX, FC, FX, Constant, Diagram, Functor, Id, Ty)
from discopy.python import Function

X, Y, Z = map(Ty, "XYZ")
TERMS = [
    FC(Constant("g", Z << Y), Constant("f", Y << X)),
    BC(Constant("f", X >> Y), Constant("g", Y >> Z)),
    FX(Constant("g", Z << Y), Constant("f", X >> Y)),
    BX(Constant("f", Y << X), Constant("g", Y >> Z))]


@pytest.mark.parametrize("term", TERMS, ids=lambda t: type(t).__name__)
def test_b81_binary_term_composes(term):
    assert (term >> Id(term.cod)).cod == term.cod
    assert (term @ X).cod == term.cod @ X


@pytest.mark.parametrize("term", TERMS, ids=lambda t: type(t).__name__)
def test_b81_binary_term_draws(term):
    assert term.to_drawing() is not None


@pytest.mark.parametrize("term", TERMS[:2], ids=lambda t: type(t).__name__)
def test_b81_binary_term_is_mapped_by_a_functor(term):
    image = Functor.id(Diagram)(term)
    assert (image.dom, image.cod) == (term.dom, term.cod)


@pytest.mark.parametrize("term", TERMS[2:], ids=lambda t: type(t).__name__)
def test_b81_crossed_term_functor_control(term):
    "Passing control: FX and BX override eval(functor) so a functor maps them."
    assert Functor.id(Diagram)(term) == term.eval()


def test_b81_nested_tree_call():
    x = cfg.Ty('x')
    f, a = cfg.Rule(x @ x, x, name='f'), cfg.Word('a', x)
    assert str(f(f, f)(a, a, a, a)) == "f(f(a, a), f(a, a))"
    nested = f(f(f, f), f)(a, a, a, a, a, a)
    assert str(nested) == "f(f(f(a, a), f(a, a)), f(a, a))"


def test_b81_to_compact_evaluates_through_python_functor():
    x, y, z = map(closed.Ty, "xyz")
    f = closed.Box("f", x @ y, z)
    F = closed.Functor(
        ob_map={x: int, y: int, z: int}, ar_map={f: lambda a, b: a - b},
        cod=Function)
    assert F(f.curry())(1)(2) == -1
    assert F(f.curry().to_compact())(1)(2) == -1
