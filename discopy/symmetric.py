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
...     space=2, path='docs/_static/symmetric/hexagons.svg', figsize=(5, 2))

.. image:: /_static/symmetric/hexagons.svg
    :align: center

Involution
==========
a.k.a. Reidemeister move 2

>>> assert Swap(x, y)[::-1] == Swap(y, x)
>>> assert Equation(Swap(x, y) >> Swap(y, x), Id(x @ y))
>>> Equation(Swap(x, y) >> Swap(y, x), Id(x @ y)).draw(
...     path='docs/_static/symmetric/inverse.svg', figsize=(3, 2))

.. image:: /_static/symmetric/inverse.svg
    :align: center

Naturality
==========

>>> naturality = Equation(
...     f @ g >> Swap(f.cod, g.cod), Swap(f.dom, g.dom) >> g @ f)
>>> assert naturality
>>> naturality.draw(
...     path='docs/_static/symmetric/naturality.svg', figsize=(3, 2))

.. image:: /_static/symmetric/naturality.svg
    :align: center

Yang-Baxter
===========
a.k.a. Reidemeister move 3

This is a special case of naturality.

>>> yang_baxter_left = Swap(x, y) @ z >> y @ Swap(x, z) >> Swap(y, z) @ x
>>> yang_baxter_right = x @ Swap(y, z) >> Swap(x, z) @ y >> z @ Swap(x, y)
>>> assert Equation(yang_baxter_left, yang_baxter_right)
>>> Equation(yang_baxter_left, yang_baxter_right).draw(
...     path='docs/_static/symmetric/yang-baxter.svg', figsize=(3, 2))

.. image:: /_static/symmetric/yang-baxter.svg
    :align: center

Permutations
============

A :class:`Permutation` is a :class:`Box` that reorders its input wires,
holding a :class:`discopy.python.finset.Permutation` as attribute. It is equal
to the diagram with just that permutation and draws as a single band of
crossing wires rather than a staircase of swaps.

>>> perm = Permutation(x @ y @ z, [2, 0, 1])
>>> assert perm.cod == z @ x @ y
>>> assert perm >> perm.dagger() == Id(x @ y @ z)
>>> assert perm @ Id(w) == Permutation(x @ y @ z @ w, [2, 0, 1, 3])
>>> assert Permutation(x @ y, [1, 0]) == Swap(x, y)

Layers
======

A symmetric :class:`Layer` puts boxes in parallel with permutations routing
the wires between them. It has the same interface ``(left, box, right, *more)``
as :class:`monoidal.Layer`, but ``left`` and ``right`` are now permutations
(an identity permutation being represented by a :class:`Ty`).

>>> layer = Layer(Permutation(x @ y, [1, 0]), f, z)
>>> assert layer.dom == x @ y @ x @ z and layer.cod == y @ x @ y @ z

Foliation
=========

Writing permutations by hand keeps swap-heavy diagrams compact: a whole
permutation occupies a single layer rather than a quadratic staircase of
swaps. For example, reversing four wires before a single layer of boxes is a
permutation layer followed by a box layer.

>>> f0, f1 = Box("f0", w, x), Box("f1", z, y)
>>> g0, g1 = Box("g0", y, z), Box("g1", x, w)
>>> reverse = Permutation(x @ y @ z @ w, [3, 2, 1, 0])
>>> diagram = reverse >> f0 @ f1 @ g0 @ g1
>>> diagram.depth()
1
>>> diagram.draw(
...     path='docs/_static/symmetric/foliation.png', figsize=(4, 4))

.. image:: /_static/symmetric/foliation.png
    :align: center
"""

from __future__ import annotations

from collections.abc import Sequence

from discopy import monoidal, balanced, traced, messages, hypergraph
from discopy.abc import SymmetricCategory
from discopy.cat import factory
from discopy.monoidal import Wire, Ty, PRO  # noqa: F401
from discopy.python import finset
from discopy.utils import factory_name, assert_isinstance


@factory
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
    trace routing) use ``from discopy.symmetric import Equation``, i.e. the
    :class:`Equation` whose :attr:`~Equation.up_to` is :attr:`to_hypergraph`.

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
    ...              path='docs/_static/symmetric/decorator.svg')

    .. image:: /_static/symmetric/decorator.svg
        :align: center

    Every variable must be used exactly once or this will raise an error.

    >>> from pytest import raises
    >>> from discopy.utils import AxiomError

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, x @ x)(lambda x: (x, x))
    >>> print(err.value)
    symmetric.Diagram has no spiders, cups or caps to draw this hypergraph.

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, Ty())(lambda x: ())
    >>> print(err.value)
    symmetric.Diagram has no spiders, cups or caps to draw this hypergraph.

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
    def permutation(cls, xs: Sequence[int], dom: monoidal.Ty = None
                    ) -> Diagram:
        """
        The diagram that encodes a given permutation as a network of swaps.

        Parameters:
            xs : A permutation, as a sequence of integers or a
                 :class:`finset.Permutation`.
            dom : A type of the same length as :code:`xs`,
                  default is :code:`PRO(len(xs))`.
        """
        xs = list(xs)
        dom = PRO(len(xs)) if dom is None else dom
        if list(range(len(dom))) != sorted(xs):
            raise ValueError(messages.WRONG_PERMUTATION.format(len(dom), xs))
        if len(dom) <= 1:
            return cls.id(dom)
        i = xs[0]
        return cls.swap(dom[:i], dom[i]) @ dom[i + 1:]\
            >> dom[i] @ cls.permutation(
                [x - 1 if x > i else x for x in xs[1:]], dom[:i] + dom[i + 1:])

    @classmethod
    def from_permutation(cls, perm: Sequence[int], dom: monoidal.Ty = None
                         ) -> Diagram:
        """
        The diagram made of a single permutation layer, or the identity if the
        permutation is the identity.

        Unlike :meth:`permutation`, this encodes the permutation natively as a
        :class:`Permutation` box rather than a network of swaps. Eliminating
        identity permutations is the responsibility of this method rather than
        the :attr:`permutation_factory`.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Diagram.from_permutation([1, 2, 0], x @ y @ z)\\
        ...     == Permutation(x @ y @ z, [1, 2, 0])
        >>> assert Diagram.from_permutation(
        ...     [0, 1, 2], x @ y @ z) == Id(x @ y @ z)
        """
        perm = finset.Permutation(perm, None if dom is None else len(dom))
        dom = PRO(len(perm)) if dom is None else dom
        if perm.is_identity:
            return cls.id(dom)
        return cls.permutation_factory(dom, perm)

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
        return hypergraph.Hypergraph[type(self).ar].from_diagram(self)

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


class Permutation(Box):
    """
    A permutation box, i.e. a :class:`Box` that reorders its input wires.

    A permutation holds a :class:`discopy.python.finset.Permutation` ``perm``
    as attribute, with the convention that output wire ``i`` comes from input
    wire ``perm[i]``, i.e. ``cod[i] == dom[perm[i]]``.

    Being a box, a permutation is equal to the diagram with just that
    permutation, just like any generator. It composes, tensors and daggers as
    a permutation: the composite of two permutations is a single permutation
    and the identity permutation is the identity diagram.

    Parameters:
        dom : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.

    Examples
    --------
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> perm = Permutation(x @ y @ z, [1, 2, 0])
    >>> assert perm.cod == y @ z @ x
    >>> assert perm.dagger() == Permutation(y @ z @ x, [2, 0, 1])
    >>> assert perm >> perm.dagger() == Id(x @ y @ z)
    """
    def __init__(self, dom: monoidal.Ty, perm):
        perm = finset.Permutation(perm, len(dom))
        if perm.is_identity:
            raise ValueError(messages.IDENTITY_PERMUTATION)
        self.perm = perm
        cod = dom[:0].tensor(*[dom[perm[i]] for i in range(len(perm))])
        super().__init__(
            f"Permutation({dom}, {list(perm)})", dom, cod, draw_as_wires=True)

    @property
    def is_identity(self) -> bool:
        """ Whether this is the identity permutation (never true by
        construction, as identity permutations are the identity diagram). """
        return self.perm.is_identity

    def to_drawing(self):
        from discopy.drawing import Drawing
        return Drawing.permutation(list(self.perm), self.dom)

    def to_swaps(self) -> Diagram:
        """The same permutation built as a network of swaps.

        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> perm = Permutation(x @ y @ z, [1, 2, 0])
        >>> assert Equation(perm.to_swaps(), perm)
        """
        return self.ar.permutation(self.perm, self.dom)

    def then(self, other=None, *others):
        if isinstance(other, Permutation) and self.cod == other.dom:
            result = self.ar.from_permutation(
                self.perm.then(other.perm), self.dom)
            return result if not others else result.then(*others)
        return super().then(other, *others)

    def tensor(self, other=None, *others):
        if other is None:
            return self
        if isinstance(other, Permutation):
            perm, dom = other.perm, other.dom
        elif isinstance(other, monoidal.Ty):
            perm, dom = finset.Permutation.id(len(other)), other
        elif isinstance(other, Diagram) and not other.inside:  # An identity.
            perm, dom = finset.Permutation.id(len(other.dom)), other.dom
        else:
            return super().tensor(other, *others)
        result = self.ar.from_permutation(
            self.perm.tensor(perm), self.dom @ dom)
        return result if not others else result.tensor(*others)

    def dagger(self) -> Permutation:
        return self.ar.from_permutation(self.perm.dagger(), self.cod)

    def __rmatmul__(self, other):
        if isinstance(other, monoidal.Ty):
            return self.ar.from_permutation(
                finset.Permutation.id(len(other)).tensor(self.perm),
                other @ self.dom)
        return super().__rmatmul__(other)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.dom!r}, {list(self.perm)})"

    def __str__(self):
        return f"Permutation({self.dom}, {list(self.perm)})"

    def __eq__(self, other):
        if isinstance(other, Swap):
            return self.dom == other.dom and tuple(self.perm) == (1, 0)
        return super().__eq__(other)

    def __hash__(self):
        if len(self.dom) == 2 and tuple(self.perm) == (1, 0):
            return hash(self.braid_factory(self.dom[:1], self.dom[1:]))
        return super().__hash__()


class Layer(monoidal.Layer):
    """
    A symmetric layer is a box in the middle of a pair of permutations, with
    more boxes and permutations to the right.

    It has the same interface ``(left, box, right, *more)`` as
    :class:`monoidal.Layer`, but now the arguments at even positions
    (``left``, ``right``, ...) are permutations rather than mere types: each
    one is either a :class:`Ty` (the identity permutation, as in
    ``monoidal.Layer``) or a non-identity :class:`Permutation` box. The
    arguments at odd positions are non-permutation boxes.

    A layer with a single argument ``Layer(perm)`` is a permutation-only layer,
    allowed as long as ``perm`` is a non-identity :class:`Permutation`.

    Parameters:
        inside : The types, permutations and boxes inside the layer, at
                 even and odd positions respectively.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> p = Permutation(x @ y, [1, 0])
    >>> layer = Layer(p, f, y)
    >>> assert layer.dom == x @ y @ x @ y and layer.cod == y @ x @ y @ y

    A permutation-only layer has a single argument and no box:

    >>> perm_layer = Layer(Permutation(x @ y, [1, 0]))
    >>> assert perm_layer.dom == x @ y and perm_layer.cod == y @ x
    >>> assert perm_layer.boxes == []
    """
    def __init__(self, *inside):
        if len(inside) % 2 == 0:
            raise ValueError(messages.LAYERS_MUST_BE_ODD)
        self.boxes_or_types = tuple(inside)
        empty = (inside[0].dom if isinstance(inside[0], Permutation)
                 else inside[0])[:0]
        name, dom, cod = "", empty, empty
        for i, x in enumerate(inside):
            if i % 2:
                assert_isinstance(x, monoidal.Box)
                if isinstance(x, Permutation):
                    raise ValueError(messages.PERMUTATION_AT_ODD_INDEX)
                dom, cod = dom @ x.dom, cod @ x.cod
                name += ("" if not name else " @ ") + str(x)
            elif isinstance(x, Permutation):
                dom, cod = dom @ x.dom, cod @ x.cod
                name += ("" if not name else " @ ") + str(x)
            else:
                assert_isinstance(x, monoidal.Ty)
                dom, cod = dom @ x, cod @ x
                name += "" if not x else ("" if not name else " @ ") + str(x)
        super(monoidal.Layer, self).__init__(name, dom, cod)

    @classmethod
    def cast(cls, box: "Box") -> Layer:
        """
        Turn a box into a layer, a permutation into a permutation-only layer.

        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> assert Layer.cast(f) == Layer(Ty(), f, Ty())
        >>> p = Permutation(x @ y, [1, 0])
        >>> assert Layer.cast(p) == Layer(p)
        """
        if isinstance(box, Permutation):
            return cls(box)
        return cls(box.dom[:0], box, box.cod[len(box.cod):])

    def dagger(self) -> Layer:
        return type(self)(*(x.dagger() for x in self))

    @property
    def boxes_and_offsets(self) -> list[tuple[monoidal.Box, int]]:
        """
        The boxes inside the layer with their offsets, as in
        :class:`monoidal.Layer`. A non-identity :class:`Permutation` has no
        offset interpretation, so this raises when the layer has one.

        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> [(str(b), i) for b, i in Layer(x, f, y).boxes_and_offsets]
        [('f', 1)]
        >>> from pytest import raises
        >>> with raises(NotImplementedError):
        ...     Layer(Permutation(x @ y, [1, 0])).boxes_and_offsets
        """
        if any(isinstance(x, Permutation) for x in self):
            raise NotImplementedError(messages.PERMUTATION_HAS_NO_OFFSET)
        return super().boxes_and_offsets


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

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return other == self
        return super().__eq__(other)

    __hash__ = Box.__hash__


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
        if isinstance(other, Permutation):
            from_permutation = getattr(self.cod, "from_permutation", None)
            if from_permutation is not None:
                return from_permutation(list(other.perm), self(other.dom))
            return self(other.to_swaps())
        return super().__call__(other)


class CMap(traced.CMap):
    category = Diagram
    require_planar = False


Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.layer_factory = Layer
Diagram.braid_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.trace_factory = Trace
Diagram.sum_factory = Sum
Id = Diagram.id


class Equation(monoidal.Equation):
    """
    The :class:`monoidal.Equation` of symmetric diagrams compared up to
    hypergraph isomorphism, i.e. up to swaps, spider fusion and trace routing.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Equation(Swap(x, y) >> Swap(y, x), Id(x @ y))
    """
    up_to = staticmethod(Diagram.to_hypergraph)
