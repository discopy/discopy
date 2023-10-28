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
    Category
    Functor

Axioms
------

>>> from discopy.drawing import Equation
>>> x, y, z, w = map(Ty, "xyzw")
>>> f, g = Box("f", x, y), Box("g", z, w)

* Triangle:

>>> assert Diagram.swap(Ty(), x) == Id(x) == Diagram.swap(x, Ty())

* Hexagon:

>>> assert Diagram.swap(x, y @ z) == Swap(x, y) @ z >> y @ Swap(x, z)
>>> assert Diagram.swap(x @ y, z) == x @ Swap(y, z) >> Swap(x, z) @ y
>>> Equation(Diagram.swap(x, y @ z), Diagram.swap(x @ y, z), symbol='').draw(
...     space=2, path='docs/_static/symmetric/hexagons.png', figsize=(5, 2))

.. image:: /_static/symmetric/hexagons.png
    :align: center

* Involution (a.k.a. Reidemeister move 2):

>>> assert Swap(x, y)[::-1] == Swap(y, x)
>>> with Diagram.hypergraph_equality:
...     assert Swap(x, y) >> Swap(y, x) == Id(x @ y)
>>> Equation(Swap(x, y) >> Swap(y, x), Id(x @ y)).draw(
...     path='docs/_static/symmetric/inverse.png', figsize=(3, 2))

.. image:: /_static/symmetric/inverse.png
    :align: center

* Naturality:

>>> naturality = Equation(
...     f @ g >> Swap(f.cod, g.cod), Swap(f.dom, g.dom) >> g @ f)
>>> with Diagram.hypergraph_equality:
...     assert naturality
>>> naturality.draw(
...     path='docs/_static/symmetric/naturality.png', figsize=(3, 2))

.. image:: /_static/symmetric/naturality.png
    :align: center

* Yang-Baxter (a.k.a. Reidemeister move 3):

>>> yang_baxter_left = Swap(x, y) @ z >> y @ Swap(x, z) >> Swap(y, z) @ x
>>> yang_baxter_right = x @ Swap(y, z) >> Swap(x, z) @ y >> z @ Swap(x, y)
>>> with Diagram.hypergraph_equality:
...     assert yang_baxter_left == yang_baxter_right
>>> Equation(yang_baxter_left, yang_baxter_right).draw(
...     path='docs/_static/symmetric/yang-baxter.png', figsize=(3, 2))

.. image:: /_static/symmetric/yang-baxter.png
    :align: center
"""

from __future__ import annotations

from contextlib import contextmanager

from discopy import monoidal, balanced, messages
from discopy.cat import factory
from discopy.monoidal import Ob, Ty, PRO  # noqa: F401
from discopy.utils import classproperty


@factory
class Diagram(balanced.Diagram):
    """
    A symmetric diagram is a balanced diagram with :class:`Swap` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ____
    Symmetric diagrams have a class property `use_hypergraph_equality`, that
    changes the behaviour of equality and hashing.
    When set to `False`, two diagrams equal if they are built from the same
    layers.
    When set to `True`, the underlying hypergraphs are used for hashing and
    equality checking.
    The default value of `use_hypergraph_equality` is `False`.

    >>> x, y = Ty("x"), Ty("y")
    >>> id_hash = hash(Id(x @ y))
    >>> assert Swap(x, y) >> Swap(y, x) != Id(x @ y)
    >>> with Diagram.hypergraph_equality:
    ...     assert Swap(x, y) >> Swap(y, x) == Id(x @ y)
    ...     assert id_hash != hash(Id(x @ y))

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
    >>> diagram.draw(draw_type_labels=False,
    ...              path='docs/_static/symmetric/decorator.png')

    .. image:: /_static/symmetric/decorator.png
        :align: center

    Every variable must be used exactly once or this will raise an error.

    >>> from pytest import raises
    >>> from discopy.utils import AxiomError

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, x @ x)(lambda x: (x, x))
    >>> print(err.value)
    symmetric.Diagram does not have copy or discard.

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, Ty())(lambda x: ())
    >>> print(err.value)
    symmetric.Diagram does not have copy or discard.


    Note
    ----
    As for :class:`discopy.balanced.Diagram`, our symmetric diagrams are traced
    by default. However now we have that the axioms for trace hold on the nose.
    """
    twist_factory = classmethod(lambda cls, dom: cls.id(dom))
    use_hypergraph_equality = False

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

    def to_hypergraph(self) -> Hypergraph:
        """ Translate a diagram into a hypergraph. """
        category = Category(self.ty_factory, self.factory)
        functor = self.hypergraph_factory.functor
        return self.hypergraph_factory[category, functor].from_diagram(self)

    def simplify(self):
        """ Simplify by translating back and forth to hypergraph. """
        return self.to_hypergraph().to_diagram()

    def _get_structure(self):
        return self.to_hypergraph() if self.use_hypergraph_equality else (
            self.inside, self.cod, self.dom)

    def __eq__(self, other):
        return isinstance(other, self.factory)\
            and self._get_structure() == other._get_structure()

    def __hash__(self):
        return hash(self._get_structure())

    @classproperty
    @contextmanager
    def hypergraph_equality(cls):
        tmp, cls.use_hypergraph_equality = cls.use_hypergraph_equality, True
        try:
            yield
        finally:
            cls.use_hypergraph_equality = tmp

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
    __ambiguous_inheritance__ = (balanced.Box, )


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
    __ambiguous_inheritance__ = (balanced.Trace, )
    __eq__, __hash__ = Diagram.__eq__, Diagram.__hash__


class Sum(balanced.Sum, Box):
    """
    A symmetric sum is a balanced sum and a symmetric box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """
    __ambiguous_inheritance__ = (balanced.Sum, )


class Category(balanced.Category):
    """
    A symmetric category is a balanced category with a method :code:`swap`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(balanced.Functor):
    """
    A symmetric functor is a monoidal functor that preserves swaps.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Swap):
            return self.cod.ar.swap(self(other.dom[0]), self(other.dom[1]))
        return super().__call__(other)


class Hypergraph(balanced.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory = Swap
Diagram.trace_factory = Trace
Diagram.sum_factory = Sum
Id = Diagram.id
