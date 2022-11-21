# -*- coding: utf-8 -*-

"""
The free closed monoidal category, i.e. with exponential objects.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Over
    Under
    Diagram
    Box
    Eval
    Curry
    Category
    Functor

Axioms
------

TODO: add axioms
"""

from __future__ import annotations

from discopy import monoidal
from discopy.cat import Category, factory
from discopy.utils import factory_name


@factory
class Ty(monoidal.Ty):
    """
    A closed type is a monoidal type that can be exponentiated.

    Parameters:
        inside (Ty) : The objects inside the type.
    """
    def __pow__(self, other: Ty) -> Ty:
        return Exp(self, other) if isinstance(other, Ty)\
            else super().__pow__(other)

    def __lshift__(self, other):
        return Over(self, other)

    def __rshift__(self, other):
        return Under(self, other)


class Exp(Ty):
    """
    A :code:`base` type to an :code:`exponent` type, called with :code:`**`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def __init__(self, base: Ty, exponent: Ty):
        self.base, self.exponent = base, exponent
        super().__init__(self)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.base, self.exponent) == (other.base, other.exponent)
        return isinstance(other, Ty) and other.inside == (self, )

    def __str__(self):
        return "({} ** {})".format(self.base, self.exponent)

    def __repr__(self):
        return factory_name(type(self)) + "({}, {})".format(
            repr(self.base), repr(self.exponent))


class Over(Exp):
    """
    A :code:`base` type over an :code:`exponent` type, called with :code:`<<`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def __str__(self):
        return "({} << {})".format(self.base, self.exponent)


class Under(Exp):
    """
    A :code:`base` type under an :code:`exponent` type, called with :code:`>>`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def __str__(self):
        return "({} >> {})".format(self.exponent, self.base)


@factory
class Diagram(monoidal.Diagram):
    """
    A closed diagram is a monoidal diagram
    with :class:`Curry` and :class:`Eval` boxes.

    Parameters:
        inside (tuple[monoidal.Layer, ...]) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    curry = lambda self, n=1, left=True: Curry(self, n, left)

    @staticmethod
    def eval(base: Ty, exponent: Ty, left=True) -> Eval:
        return Eval(base << exponent if left else exponent >> base)

    def uncurry(self: Diagram, left=True) -> Diagram:
        base, exponent = self.cod.base, self.cod.exponent
        return self @ exponent >> Eval(base << exponent) if left\
            else exponent @ self >> Eval(exponent >> base)


class Box(monoidal.Box, Diagram):
    """
    A closed box is a monoidal box in a closed diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Eval(Box):
    """
    The evaluation of an exponent type.

    Parameters:
        x : The exponent type to evaluate.
    """
    def __init__(self, x: Exp):
        self.base, self.exponent = x.base, x.exponent
        self.left = isinstance(x, Over)
        dom, cod = (x @ self.exponent, self.base) if self.left\
            else (self.exponent @ x, self.base)
        super().__init__("Eval" + str(x), dom, cod)

class Curry(Box):
    """
    The currying of a closed diagram.

    Parameters:
        diagram : The diagram to curry.
        n : The number of atomic types to curry.
        left : Whether to curry on the left or right.
    """
    def __init__(self, diagram: Diagram, n=1, left=True):
        self.diagram, self.n, self.left = diagram, n, left
        name = "Curry({}, {}, {})".format(diagram, n, left)
        if left:
            dom = diagram.dom[:len(diagram.dom) - n]
            cod = diagram.cod << diagram.dom[len(diagram.dom) - n:]
        else:
            dom, cod = diagram.dom[n:], diagram.dom[:n] >> diagram.cod
        super().__init__(name, dom, cod)


Diagram.over, Diagram.under, Diagram.exp = map(staticmethod, (Over, Under, Exp))


class Category(monoidal.Category):
    """
    A closed category is a monoidal category
    with methods :code:`eval` and :code:`curry`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = Ty, Diagram


class Functor(monoidal.Functor):
    """
    A closed functor is a monoidal functor
    that preserves evaluation and currying.

    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        for cls, attr in [(Over, "over"), (Under, "under"), (Exp, "exp")]:
            if isinstance(other, cls):
                method = getattr(self.cod.ar, attr)
                return method(self(other.base), self(other.exponent))
        if isinstance(other, Curry):
            return self.cod.ar.curry(
                self(other.diagram), len(self(other.cod.exponent)), other.left)
        if isinstance(other, Eval):
            return self.cod.ar.eval(
                self(other.base), self(other.exponent), other.left)
        return super().__call__(other)
