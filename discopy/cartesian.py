# -*- coding: utf-8 -*-

"""
The free cartesian category, i.e. diagrams with copy and discard.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Copy
    Category
    Functor


Axioms
------

Coherence holds on the nose.

>>> x, y = Ty('x'), Ty('y')
>>> multicopy = Copy(x) @ Copy(y) >> Id(x) @ Swap(x, y) @ Id(y)
>>> assert Diagram.copy(x @ y) == multicopy

The axioms of cartesian categories cannot be checked in DisCoPy, we can
draw them and check whether they hold for a given ``Functor``.

>>> copy_l = Copy(x) >> Copy(x) @ Id(x)
>>> copy_r = Copy(x) >> Id(x) @ Copy(x)

>>> from discopy.drawing import Equation
>>> Equation(copy_l, copy_r, symbol="=").draw(
...     path="docs/_static/cartesian/associativity.png")

.. image:: /_static/cartesian/associativity.png

>>> delete = lambda x: Copy(x, n=0)
>>> counit_l = Copy(x) >> delete(x) @ Id(x)
>>> counit_r = Copy(x) >> Id(x) @ delete(x)
>>> Equation(counit_l, Id(x), counit_r, symbol="=").draw(
...     path="docs/_static/cartesian/counit.png")

.. image:: /_static/cartesian/counit.png
"""

from __future__ import annotations

from discopy import symmetric, monoidal, frobenius
from discopy.cat import factory
from discopy.monoidal import Ty, assert_isatomic


@factory
class Diagram(symmetric.Diagram):
    """
    A cartesian diagram is a symmetric diagram with :class:`Copy` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        cls.spider_factory = lambda _a, b, x, _p: Copy(x, b)
        result = frobenius.Diagram.spiders.__func__(cls, 1, n, x)
        del cls.spider_factory
        return result


class Box(symmetric.Box, Diagram):
    """
    A cartesian box is a symmetric box in a cartesian diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, )


class Swap(symmetric.Swap, Box):
    """
    Symmetric swap in a cartesian diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (symmetric.Swap, )


Diagram.braid_factory = Swap


class Copy(Box):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type to copy.
        n : The number of copies.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        super().__init__(name=f"Copy({x}, {n})", dom=x, cod=x ** n,
                         draw_as_spider=True, color="black", drawing_name="")


class Category(symmetric.Category):
    """
    A cartesian category is a symmetric category with a method :code:`copy`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = monoidal.Ty, Diagram


class Functor(symmetric.Functor):
    """
    A cartesian functor is a symmetric functor that preserves copies.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.

    Example
    -------

    We build a functor into python functions.

    >>> x = Ty('x')
    >>> add = Box('add', x @ x, x)
    >>> from discopy import python
    >>> F = Functor({x: int}, {add: lambda a, b: a + b},
    ...     Category(python.Ty, python.Function))
    >>> copy = Copy(x)
    >>> bialgebra_l = copy @ copy >> Id(x) @ Swap(x, x) @ Id(x) >> add @ add
    >>> bialgebra_r = add >> copy
    >>> assert F(bialgebra_l)(54, 46) == F(bialgebra_r)(54, 46)

    >>> from discopy.drawing import Equation
    >>> Equation(bialgebra_l, bialgebra_r, symbol="=").draw(
    ...     path="docs/_static/cartesian/bialgebra.png")

    .. image:: /_static/cartesian/bialgebra.png
    """
    dom = cod = Category(monoidal.Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Copy):
            return self.cod.ar.copy(self(other.dom), len(other.cod))
        return super().__call__(other)


Id = Diagram.id
