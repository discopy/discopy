# -*- coding: utf-8 -*-

"""
The free symmetric category, i.e. diagrams with swaps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Permutation
    Layer
    Diagram
    Box
    Swap
    Sum
    Functor

Axioms
------

>>> from discopy.drawing import Equation
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
>>> with Diagram.hypergraph_equality:
...     assert Swap(x, y) >> Swap(y, x) == Id(x @ y)
>>> Equation(Swap(x, y) >> Swap(y, x), Id(x @ y)).draw(
...     path='docs/_static/symmetric/inverse.png', figsize=(3, 2))

.. image:: /_static/symmetric/inverse.png
    :align: center

Naturality
==========

>>> naturality = Equation(
...     f @ g >> Swap(f.cod, g.cod), Swap(f.dom, g.dom) >> g @ f)
>>> with Diagram.hypergraph_equality:
...     assert naturality
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
>>> with Diagram.hypergraph_equality:
...     assert yang_baxter_left == yang_baxter_right
>>> Equation(yang_baxter_left, yang_baxter_right).draw(
...     path='docs/_static/symmetric/yang-baxter.png', figsize=(3, 2))

.. image:: /_static/symmetric/yang-baxter.png
    :align: center

Permutations
============

>>> perm = Permutation([2, 0, 1], x @ y @ z)
>>> assert perm.cod == z @ x @ y
>>> assert perm.then(perm.dagger()) == Permutation.id(x @ y @ z)
>>> assert perm @ Permutation.id(w) == Permutation([2, 0, 1, 3], x @ y @ z @ w)

Layers
======

>>> p = Permutation([1, 0], x @ y)
>>> layer = Layer(p, f, Permutation.id(y))
>>> assert layer.dom == x @ y @ x @ y
>>> assert layer.cod == y @ x @ y @ y
>>> layer2 = Layer(Permutation.id(z), g, Permutation.id(w))
>>> combined = layer @ layer2
>>> assert combined.dom == layer.dom @ layer2.dom
"""

from __future__ import annotations

from contextlib import contextmanager

from discopy import monoidal, balanced, traced, messages
from discopy.abc import SymmetricCategory
from discopy.cat import Arrow, ar_factory
from discopy.monoidal import Ob, Ty, PRO  # noqa: F401
from discopy.utils import (
    factory_name, classproperty, assert_isinstance, AxiomError)


class Permutation:
    """
    A permutation of a type, forming a dagger monoidal category.

    A permutation is stored as a list of integers ``inside`` with a domain
    type ``dom``. The convention is that ``inside[i] = j`` means output wire
    ``i`` comes from input wire ``j``.

    Parameters:
        inside : A list of integers encoding the permutation.
        dom : The domain type, default is ``PRO(len(inside))``.

    Examples
    --------
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> perm = Permutation([1, 2, 0], x @ y @ z)
    >>> assert perm.cod == y @ z @ x
    >>> assert perm.dagger() == Permutation([2, 0, 1], y @ z @ x)
    >>> assert perm.then(perm.dagger()) == Permutation.id(x @ y @ z)
    """
    ty_factory = Ty

    def __init__(self, inside: list[int], dom: monoidal.Ty = None):
        inside = list(inside)
        dom = PRO(len(inside)) if dom is None else dom
        if sorted(inside) != list(range(len(dom))):
            raise ValueError(
                messages.WRONG_PERMUTATION.format(len(dom), inside))
        if len(inside) != len(dom):
            raise ValueError(
                messages.WRONG_PERMUTATION.format(len(dom), inside))
        self.inside = inside
        self.dom = dom

    @property
    def cod(self) -> monoidal.Ty:
        """The codomain: ``cod[i] = dom[inside[i]]``."""
        if not self.inside:
            return self.dom
        return self.dom[:0].tensor(
            *[self.dom[j] for j in self.inside])

    @property
    def is_identity(self) -> bool:
        return self.inside == list(range(len(self.dom)))

    @classmethod
    def id(cls, dom: monoidal.Ty = None) -> Permutation:
        """The identity permutation on a given type."""
        dom = cls.ty_factory() if dom is None else dom
        return cls(list(range(len(dom))), dom)

    def then(self, other: Permutation) -> Permutation:
        """
        Sequential composition:
        ``(self >> other).inside[i] = other.inside[self.inside[i]]``.
        """
        if self.cod != other.dom:
            raise AxiomError(
                f"Permutation {self} does not compose with {other}: "
                f"{self.cod} != {other.dom}")
        composed = [other.inside[self.inside[i]]
                    for i in range(len(self.inside))]
        return type(self)(composed, self.dom)

    def tensor(self, other: Permutation) -> Permutation:
        """
        Parallel composition: place ``self`` and ``other`` side by side.
        """
        n = len(self.inside)
        inside = self.inside + [j + n for j in other.inside]
        dom = self.dom @ other.dom
        return type(self)(inside, dom)

    def dagger(self) -> Permutation:
        """
        The inverse permutation.

        If ``self.inside[i] = j``, then ``self.dagger().inside[j] = i``.
        """
        n = len(self.inside)
        inv = [0] * n
        for i, j in enumerate(self.inside):
            inv[j] = i
        return type(self)(inv, self.cod)

    def __matmul__(self, other):
        if isinstance(other, Permutation):
            return self.tensor(other)
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, Permutation):
            return self.then(other)
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, Permutation)\
            and self.inside == other.inside and self.dom == other.dom

    def __hash__(self):
        return hash((tuple(self.inside), self.dom))

    def __repr__(self):
        return f"{factory_name(type(self))}"\
            f"({self.inside}, {repr(self.dom)})"

    def __str__(self):
        if self.is_identity:
            return f"Perm.id({self.dom})"
        return f"Perm({self.inside}, {self.dom})"

    def __len__(self):
        return len(self.inside)

    def whisker(self, left: monoidal.Ty = None,
                right: monoidal.Ty = None) -> Permutation:
        """
        Whisker this permutation with identity permutations.

        ``left @ self @ right`` as permutations.
        """
        result = self
        if left is not None and len(left) > 0:
            result = type(self).id(left).tensor(result)
        if right is not None and len(right) > 0:
            result = result.tensor(type(self).id(right))
        return result


class Layer:
    """
    A symmetric layer is an alternating sequence
    ``(perm, box, perm, ..., box, perm)`` representing boxes in parallel
    with permutations routing wires between them.

    The layer represents the parallel composite
    ``perm0 @ box0 @ perm1 @ ... @ boxN @ permN+1``.

    Parameters:
        *inside : Alternating :class:`Permutation` and :class:`Box` objects,
                  with permutations at even positions and boxes at odd.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> p = Permutation([1, 0], x @ y)
    >>> layer = Layer(p, f, Permutation.id(y))
    >>> assert layer.dom == x @ y @ x @ y
    >>> assert layer.cod == y @ x @ y @ y
    """
    def __init__(self, *inside):
        if len(inside) < 3 or len(inside) % 2 == 0:
            raise ValueError(
                "Layer needs an odd number of elements (>= 3).")
        self.inside = inside

    @property
    def perms(self) -> tuple:
        """The permutations at even positions."""
        return tuple(self.inside[::2])

    @property
    def boxes(self) -> tuple:
        """The boxes at odd positions."""
        return tuple(self.inside[1::2])

    @property
    def dom(self) -> monoidal.Ty:
        ty = self.inside[0].dom[:0]
        for x in self.inside:
            ty = ty @ x.dom
        return ty

    @property
    def cod(self) -> monoidal.Ty:
        ty = self.inside[0].dom[:0]
        for x in self.inside:
            ty = ty @ x.cod
        return ty

    def tensor(self, other: Layer) -> Layer:
        """
        Parallel composition: merge boundary permutations.

        ``Layer(p0, b0, p1) @ Layer(q0, c0, q1) == Layer(p0, b0, p1 @ q0, c0, q1)``
        """
        *head, last_perm = self.inside
        first_perm, *tail = other.inside
        return type(self)(*head, last_perm @ first_perm, *tail)

    def dagger(self) -> Layer:
        """Dagger each component (parallel, not reversed)."""
        return type(self)(*(x.dagger() for x in self.inside))

    def __matmul__(self, other):
        if isinstance(other, Layer):
            return self.tensor(other)
        if isinstance(other, Permutation):
            *head, last = self.inside
            return type(self)(*head, last @ other)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, Permutation):
            first, *tail = self.inside
            return type(self)(other @ first, *tail)
        return NotImplemented

    def to_diagram(self) -> Diagram:
        """Expand to a :class:`Diagram` by converting permutations to swaps."""
        result = None
        for x in self.inside:
            if isinstance(x, Permutation):
                d = Diagram.permutation(x.inside, x.dom)
            else:
                layer = Diagram.layer_factory.cast(x)
                d = Diagram((layer,), x.dom, x.cod, _scan=False)
            result = d if result is None else result @ d
        return result

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.inside == other.inside

    def __hash__(self):
        return hash(self.inside)

    def __repr__(self):
        return f"{factory_name(type(self))}"\
            f"({', '.join(map(repr, self.inside))})"

    def __str__(self):
        return ' @ '.join(map(str, self.inside))


@ar_factory
class Diagram(balanced.Diagram, SymmetricCategory):
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
    >>> diagram.draw(wire_labels=False,
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

    @classmethod
    def from_perm(cls, perm: Permutation) -> Diagram:
        """
        Create a diagram from a :class:`Permutation`.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> perm = Permutation([2, 0, 1], x @ y @ z)
        >>> d = Diagram.from_perm(perm)
        >>> assert d.dom == x @ y @ z and d.cod == z @ x @ y
        """
        if perm.is_identity:
            return cls.id(perm.dom)
        return cls.permutation(perm.inside, perm.dom)

    @classmethod
    def from_layer(cls, layer: Layer) -> Diagram:
        """
        Create a diagram from a symmetric :class:`Layer`.

        Examples
        --------
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> p = Permutation([1, 0], x @ y)
        >>> layer = Layer(p, f, Permutation.id(y))
        >>> d = Diagram.from_layer(layer)
        >>> assert d.dom == layer.dom and d.cod == layer.cod
        """
        return layer.to_diagram()

    def to_hypergraph(self) -> Hypergraph:
        """ Translate a diagram into a hypergraph. """
        return self.hypergraph_factory.from_diagram(self)

    def simplify(self):
        """ Simplify by translating back and forth to hypergraph. """
        return self.to_hypergraph().to_diagram()

    def _get_structure(self):
        return self.to_hypergraph() if self.use_hypergraph_equality else (
            self.inside, self.cod, self.dom)

    def __eq__(self, other):
        return isinstance(other, self.ar)\
            and self._get_structure() == other._get_structure()

    def __hash__(self):
        if self.use_hypergraph_equality:
            return hash(self._get_structure())
        return hash(repr(self))

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

    def __hash__(self):
        if self.use_hypergraph_equality:
            return hash(self.to_hypergraph())
        return hash(Arrow.__repr__(self))


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

    def to_perm(self) -> Permutation:
        """Convert this swap to a :class:`Permutation`."""
        return Permutation([1, 0], self.dom)


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
    __eq__, __hash__ = Diagram.__eq__, Diagram.__hash__

    def _get_structure(self):
        return super()._get_structure() if self.use_hypergraph_equality else (
            type(self), self.dom, self.cod, self.arg._get_structure())


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
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
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
