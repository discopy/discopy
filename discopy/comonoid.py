# -*- coding: utf-8 -*-

"""
The free symmetric category with a supply of (co)commutative (co)monoid,
also called copy-discard category, see :cite:t:`FritzLiang23`.

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

>>> from discopy.drawing import Equation
>>> x, y, z = map(Ty, "xyz")

>>> copy, merge = Copy(x), Merge(x)
>>> unit, delete = Merge(x, n=0), Copy(x, n=0)

* Commutative monoid:

>>> unitality = Equation(unit @ x >> merge, Id(x), x @ unit >> merge)
>>> associativity = Equation(merge @ x >> merge, x @ merge >> merge)
>>> commutativity = Equation(Swap(x, x) >> merge, merge)
>>> assert unitality and associativity and commutativity
>>> Equation(unitality, associativity, commutativity, symbol='').draw(
...     path="docs/_static/frobenius/monoid.png")

.. image:: /_static/frobenius/monoid.png
    :align: center

* Cocommutative comonoid:

>>> counitality = Equation(copy >> delete @ x, Id(x), copy >> x @ delete)
>>> coassociativity = Equation(copy >> copy @ x, copy >> x @ copy)
>>> cocommutativity = Equation(copy >> Swap(x, x), copy)
>>> assert counitality and coassociativity and cocommutativity
>>> Equation(counitality, coassociativity, cocommutativity, symbol='').draw(
...     path="docs/_static/frobenius/comonoid.png")

.. image:: /_static/frobenius/comonoid.png
    :align: center

* Coherence:

>>> assert Diagram.copy(x @ x, n=0) == delete @ delete
>>> assert Diagram.copy(x @ x)\\
...     == copy @ copy >> x @ Swap(x, x) @ x
>>> assert Diagram.merge(x @ x, n=0) == unit @ unit
>>> assert Diagram.merge(x @ x)\\
...     == x @ Swap(x, x) @ x >> merge @ merge
"""

from __future__ import annotations

from discopy import symmetric, monoidal, hypergraph
from discopy.cat import factory
from discopy.monoidal import Ty, assert_isatomic


@factory
class Diagram(symmetric.Diagram):
    """
    A comonoid diagram is a symmetric diagram with :class:`Copy` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    We can create arbitrary comonoid diagrams with the standard notation for
    Python functions.

    >>> x = Ty('x')
    >>> f = Box('f', x, x)

    >>> copy_then_apply = Diagram[x, x @ x](
    ...     lambda x: (f(x), f(x)))

    >>> @Diagram[x, x @ x]
    ... def apply_then_copy(x):
    ...     y = f(x)
    ...     return x, x

    >>> from discopy.drawing import Equation
    >>> Equation(copy_then_apply, apply_then_copy, symbol="$\\\\neq$").draw(
    ...     path="docs/_static/comonoid/copy_and_apply.png")

    .. image:: /_static/comonoid/copy_and_apply.png
    """
    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or 1 not in (n_legs_in, n_legs_out):
            raise ValueError
        return cls.copy_factory(typ, n_legs_out) if n_legs_in == 1\
            else cls.merge_factory(typ, n_legs_in)

    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(cls, 1, n, x)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Merge :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        return cls.copy(x, n).dagger()


class Box(symmetric.Box, Diagram):
    """
    A comonoid box is a symmetric box in a comonoid diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, )


class Swap(symmetric.Swap, Box):
    """
    Symmetric swap in a comonoid diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (symmetric.Swap, )


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

    def dagger(self) -> Merge:
        return Merge(self.dom, len(self.cod))


class Merge(Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        super().__init__(name=f"Merge({x}, {n})", dom=x ** n, cod=x,
                         draw_as_spider=True, color="black", drawing_name="")

    def dagger(self) -> Merge:
        return Copy(self.cod, len(self.dom))


class Category(symmetric.Category):
    """
    A comonoid category is a symmetric category with a method :code:`copy`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor):
    """
    A comonoid functor is a symmetric functor that preserves copies.

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
    ...             cod=Category(python.Ty, python.Function))
    >>> copy = Copy(x)
    >>> bialgebra_l = copy @ copy >> Id(x) @ Swap(x, x) @ Id(x) >> add @ add
    >>> bialgebra_r = add >> copy
    >>> assert F(bialgebra_l)(54, 46) == F(bialgebra_r)(54, 46)

    >>> from discopy.drawing import Equation
    >>> Equation(bialgebra_l, bialgebra_r, symbol="=").draw(
    ...     path="docs/_static/comonoid/bialgebra.png")

    .. image:: /_static/comonoid/bialgebra.png
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Copy):
            return self.cod.ar.copy(self(other.dom), len(other.cod))
        if isinstance(other, Merge):
            return self.cod.ar.merge(self(other.cod), len(other.dom))
        return super().__call__(other)


class Hypergraph(hypergraph.Hypergraph):
    category, functor = Category, Functor

    def to_diagram(self, make_progressive_first=True) -> Diagram:
        return super().to_diagram(
            make_progressive_first=make_progressive_first)


Diagram.hypergraph_factory = Hypergraph
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.braid_factory = Swap
Id = Diagram.id
