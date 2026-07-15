# -*- coding: utf-8 -*-

"""
The free symmetric category, i.e. diagrams with swaps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Sum
    Functor

Axioms
------

>>> from discopy.symmetric import Equation
>>> x, y, z, w = map(Ty, "xyzw")
>>> f, g = Box("f", x, y), Box("g", z, w)

Triangle
========

>>> assert Diagram.swap(Ty(), x) == Id(x) == Diagram.swap(x, Ty())

Hexagon
=======

>>> assert Diagram.swap(x, y @ z) == Swap(x, y) @ z >> y @ Swap(x, z)
>>> assert Diagram.swap(x @ y, z) == x @ Swap(y, z) >> Swap(x, z) @ y
>>> Equation(Diagram.swap(x, y @ z), Diagram.swap(x @ y, z), symbol='').draw(
...     space=2, path='docs/_static/symmetric/hexagons.png', figsize=(5, 2))

.. image:: /_static/symmetric/hexagons.png
    :align: center

Involution
==========
a.k.a. Reidemeister move 2

>>> assert Swap(x, y)[::-1] == Swap(y, x)
>>> assert Equation(Swap(x, y) >> Swap(y, x), Id(x @ y))
>>> Equation(Swap(x, y) >> Swap(y, x), Id(x @ y)).draw(
...     path='docs/_static/symmetric/inverse.png', figsize=(3, 2))

.. image:: /_static/symmetric/inverse.png
    :align: center

Naturality
==========

>>> naturality = Equation(
...     f @ g >> Swap(f.cod, g.cod), Swap(f.dom, g.dom) >> g @ f)
>>> assert naturality
>>> naturality.draw(
...     path='docs/_static/symmetric/naturality.png', figsize=(3, 2))

.. image:: /_static/symmetric/naturality.png
    :align: center

Yang-Baxter
===========
a.k.a. Reidemeister move 3

This is a special case of naturality.

>>> yang_baxter_left = Swap(x, y) @ z >> y @ Swap(x, z) >> Swap(y, z) @ x
>>> yang_baxter_right = x @ Swap(y, z) >> Swap(x, z) @ y >> z @ Swap(x, y)
>>> assert Equation(yang_baxter_left, yang_baxter_right)
>>> Equation(yang_baxter_left, yang_baxter_right).draw(
...     path='docs/_static/symmetric/yang-baxter.png', figsize=(3, 2))

.. image:: /_static/symmetric/yang-baxter.png
    :align: center
"""

from __future__ import annotations

from discopy import monoidal, balanced, traced, messages
from discopy.abc import SymmetricCategory
from discopy.utils import ar_factory
from discopy.monoidal import Ob, Ty, PRO  # noqa: F401


class _ToHypergraph:
    """
    Descriptor exposing the diagram-to-hypergraph functor.

    Accessed on the class, ``Diagram.to_hypergraph`` is the :class:`Functor`
    itself, so that e.g. ``Diagram.to_hypergraph.quotient`` returns an
    :class:`Equation`.  Accessed on an instance, ``diagram.to_hypergraph()``
    applies the functor to the diagram, i.e. translates it into a hypergraph.

    The functor is built once per class rather than on the fly, from the
    class' :attr:`hypergraph_factory`.
    """
    _cache: dict = {}

    @classmethod
    def functor(cls, diagram_cls):
        try:
            return cls._cache[diagram_cls]
        except KeyError:
            factory = diagram_cls.hypergraph_factory
            functor = factory.functor(
                ob_map=lambda typ: typ, ar_map=factory.from_box,
                dom=diagram_cls, cod=factory)
            cls._cache[diagram_cls] = functor
            return functor

    def __get__(self, obj, objtype=None):
        functor = self.functor(objtype if obj is None else type(obj))
        return functor if obj is None else lambda: functor(obj)


@ar_factory
class Diagram(balanced.Diagram, SymmetricCategory):
    """
    A symmetric diagram is a balanced diagram with :class:`Swap` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    Equality and hashing of symmetric diagrams is always syntactic: two
    diagrams are equal if and only if they are built from the same layers.
    To compare diagrams up to hypergraph isomorphism (swaps, spider fusion,
    trace routing) use the :class:`Equation` of :attr:`to_hypergraph`, e.g.
    ``from discopy.symmetric import Equation`` or, equivalently,
    ``Equation = Diagram.to_hypergraph.quotient``.

    >>> x, y = Ty("x"), Ty("y")
    >>> a = Swap(x, y) >> Swap(y, x)
    >>> assert a != Id(x @ y)
    >>> assert Equation(a, Id(x @ y))

    Note
    ----
    Symmetric diagrams can be defined using the standard syntax for functions.

    >>> x = Ty('x')
    >>> f = Box('f', x @ x, x)
    >>> g = Box('g', x, x @ x)

    >>> @Diagram.from_callable(x @ x @ x, x @ x @ x)
    ... def diagram(x0, x1, x2):
    ...     x3 = f(x2, x0)
    ...     x4, x5 = g(x1)
    ...     return x5, x3, x4
    >>> diagram.draw(wire_labels=False,
    ...              path='docs/_static/symmetric/decorator.png')

    .. image:: /_static/symmetric/decorator.png
        :align: center

    Every variable must be used exactly once or this will raise an error.

    >>> from pytest import raises

    >>> with raises(AttributeError) as err:
    ...     Diagram.from_callable(x, x @ x)(lambda x: (x, x))
    >>> print(err.value)
    type object 'Diagram' has no attribute 'spider_factory'

    >>> with raises(AttributeError) as err:
    ...     Diagram.from_callable(x, Ty())(lambda x: ())
    >>> print(err.value)
    type object 'Diagram' has no attribute 'spider_factory'

    Note
    ----
    As for :class:`discopy.balanced.Diagram`, our symmetric diagrams are traced
    by default. However now we have that the axioms for trace hold on the nose.
    """
    twist_factory = classmethod(lambda cls, dom: cls.id(dom))

    @classmethod
    def swap(cls, left: monoidal.Ty, right: monoidal.Ty) -> Diagram:
        """
        The diagram that swaps the ``left`` and ``right`` wires.

        Parameters:
            left : The type at the top left and bottom right.
            right : The type at the top right and bottom left.

        Note
        ----
        This calls :func:`balanced.hexagon` and :attr:`braid_factory`.
        """
        return cls.braid(left, right)

    @classmethod
    def permutation(cls, xs: list[int], dom: monoidal.Ty = None) -> Diagram:
        """
        The diagram that encodes a given permutation.

        Parameters:
            xs : A list of integers representing a permutation.
            dom : A type of the same length as :code:`permutation`,
                  default is :code:`PRO(len(permutation))`.
        """
        dom = PRO(len(xs)) if dom is None else dom
        if list(range(len(dom))) != sorted(xs):
            raise ValueError(messages.WRONG_PERMUTATION.format(len(dom), xs))
        if len(dom) <= 1:
            return cls.id(dom)
        i = xs[0]
        return cls.swap(dom[:i], dom[i]) @ dom[i + 1:]\
            >> dom[i] @ cls.permutation(
                [x - 1 if x > i else x for x in xs[1:]], dom[:i] + dom[i + 1:])

    def permute(self, *xs: int) -> Diagram:
        """
        Post-compose with a permutation.

        Parameters:
            xs : A list of integers representing a permutation.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Id(x @ y @ z).permute(2, 0, 1).cod == z @ x @ y
        """
        return self >> self.permutation(list(xs), self.cod)

    #: The functor from diagrams to hypergraphs, see :class:`_ToHypergraph`.
    #: Accessed on an instance it is callable, i.e. ``diagram
    #: .to_hypergraph()`` translates the diagram into a hypergraph; accessed on
    #: the class it is the :class:`Functor` itself, so it combines with
    #: :meth:`Functor.quotient` to compare diagrams up to hypergraph iso.
    to_hypergraph = _ToHypergraph()

    def simplify(self):
        """ Simplify by translating back and forth to hypergraph. """
        return self.to_hypergraph().to_diagram()

    def depth(self):
        """
        The depth of a symmetric diagram.

        Examples
        --------
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> assert Id(x).depth() == Id().depth() == 0
        >>> assert f.depth() == (f @ f).depth() == 1
        >>> assert (f @ f >> Swap(x, x)).depth() == 1
        >>> assert (f >> f).depth() == 2 and (f >> f >> f).depth() == 3
        """
        return self.to_hypergraph().depth()


class Box(balanced.Box, Diagram):
    """
    A symmetric box is a balanced box in a symmetric diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Swap(balanced.Braid, Box):
    """
    The swap of atomic types :code:`left` and :code:`right`.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.

    Important
    ---------
    :class:`Swap` is only defined for atomic types (i.e. of length 1).
    For complex types, use :meth:`Diagram.swap` instead.
    """
    def __init__(self, left, right):
        balanced.Braid.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod,
                     draw_as_wires=True, draw_as_braid=False)

    def dagger(self):
        return type(self)(self.right, self.left)


class Trace(balanced.Trace, Box):
    """
    A trace in a symmetric category.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
    """


class Sum(balanced.Sum, Box):
    """
    A symmetric sum is a balanced sum and a symmetric box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(balanced.Functor):
    """
    A symmetric functor is a monoidal functor that preserves swaps.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Swap):
            return self.cod.swap(self(other.dom[0]), self(other.dom[1]))
        return super().__call__(other)


class Hypergraph(balanced.Hypergraph):
    functor = Functor


class CMap(traced.CMap):
    functor = Functor
    require_planar = False


Diagram.hypergraph_factory = Hypergraph
Diagram.map_factory = CMap
Diagram.braid_factory = Swap
Diagram.trace_factory = Trace
Diagram.sum_factory = Sum
Id = Diagram.id

#: The :class:`Equation` of symmetric diagrams compared up to hypergraph
#: isomorphism, i.e. ``Equation = Diagram.to_hypergraph.quotient``.  Use it as
#: ``from discopy.symmetric import Equation`` to test axioms that only hold up
#: to swaps, spider fusion and trace routing.
Equation = Diagram.to_hypergraph.quotient
