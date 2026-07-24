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
holding a :class:`discopy.python.finset.Permutation` as attribute. In a
:class:`Layer`, permutations are structural routing components rather than
generators. They draw as a single band of crossing wires rather than a
staircase of swaps.

>>> perm = Permutation(x @ y @ z, [2, 0, 1])
>>> assert perm.cod == z @ x @ y
>>> assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
>>> assert perm @ Id(w) == Permutation(
...     x @ y @ z @ w, [2, 0, 1, 3])
>>> assert Permutation(x @ y, [1, 0]) != Swap(x, y)
>>> assert Equation(Permutation(x @ y, [1, 0]), Swap(x, y))

Layers
======

A symmetric :class:`Layer` alternates permutations and generators. Types
passed at even positions are immediately normalised to identity permutations,
so every stored even component has the same representation.

>>> layer = Layer(x, f, y)
>>> assert all(isinstance(p, Permutation) for p in layer[::2])
>>> assert all(p.is_identity for p in layer[::2])

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

from discopy import cat, monoidal, balanced, traced, messages, hypergraph
from discopy.abc import SymmetricCategory
from discopy.cat import factory
from discopy.monoidal import Wire, Ty, PRO  # noqa: F401
from discopy.python import finset
from discopy.utils import (
    AxiomError, assert_iscomposable, assert_isinstance, factory_name,
    from_tree)


class Layer(monoidal.Layer):
    """
    A tensor product alternating structural permutations and generators.

    Every even component is stored as a :class:`Permutation`, including
    identities. Passing a :class:`Ty` at an even position is convenient input
    syntax which is normalised immediately. Odd components are generators;
    in this first iteration :class:`Swap` remains such a generator and is
    distinct from the permutation ``[1, 0]``.

    Parameters:
        inside : An odd number of alternating permutations and generators.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, y)
    >>> layer = Layer(x, f, y)
    >>> assert all(isinstance(p, Permutation) for p in layer[::2])
    >>> assert layer.boxes == [f]
    >>> assert Layer(Permutation(x @ y, [1, 0])).boxes == []
    """
    def __init__(self, *inside):
        if not inside or not len(inside) % 2:
            raise ValueError(messages.LAYERS_MUST_BE_ODD)
        for box in inside[1::2]:
            assert_isinstance(box, monoidal.Box)
            if isinstance(box, Permutation):
                raise ValueError(messages.PERMUTATION_AT_ODD_INDEX)
        factory_ = type(inside[0]) if len(inside) == 1\
            and isinstance(inside[0], Permutation)\
            else inside[1].ar.permutation_factory if len(inside) > 1\
            else Permutation
        if len(inside) > 1 and not isinstance(inside[1].dom, factory_.ob):
            factory_ = Permutation
        normalised = []
        for i, value in enumerate(inside):
            if i % 2:
                normalised.append(value)
                continue
            if isinstance(value, monoidal.Ty):
                value = factory_(value, finset.Permutation.id(len(value)))
            else:
                assert_isinstance(value, Permutation)
                if type(value) is not factory_:
                    value = factory_(value.dom, value.perm)
            normalised.append(value)
        self.boxes_or_types = tuple(normalised)
        empty = normalised[0].dom[:0]
        dom = empty.tensor(*(value.dom for value in normalised))
        cod = empty.tensor(*(value.cod for value in normalised))
        names = [
            str(value.dom) if i % 2 == 0 and value.is_identity
            else str(value)
            for i, value in enumerate(normalised)
            if i % 2 or not value.is_identity or value.dom]
        cat.Box.__init__(self, " @ ".join(names), dom, cod)

    @property
    def boxes_and_types(self):
        """ The ordinary types underlying the structural permutations. """
        if any(not permutation.is_identity
               for permutation in self.permutations):
            raise NotImplementedError(messages.PERMUTATION_HAS_NO_TYPE_SLOT)
        return tuple(
            value.dom if i % 2 == 0 else value
            for i, value in enumerate(self))

    @property
    def permutations(self) -> list[Permutation]:
        """ The structural routing components at even positions. """
        return list(self.boxes_or_types[::2])

    @property
    def is_permutation(self) -> bool:
        """ Whether this is a permutation-only layer. """
        return len(self.boxes_or_types) == 1

    @property
    def permutation(self) -> Permutation | None:
        """ The sole permutation when :attr:`is_permutation` holds. """
        return self.boxes_or_types[0] if self.is_permutation else None

    @property
    def is_generator(self) -> bool:
        if self.is_permutation:
            return False
        if len(self.boxes_or_types) != 3:
            return False
        left, _, right = self.boxes_or_types
        return left.is_identity and right.is_identity\
            and not left.dom and not right.dom

    @property
    def generator(self) -> Box | None:
        return self.boxes_or_types[1] if self.is_generator else None

    @classmethod
    def cast(cls, box: Box) -> Layer:
        """ Turn a generator or permutation into a uniform layer. """
        if isinstance(box, Permutation):
            return cls(box)
        return cls(box.dom[:0], box, box.cod[len(box.cod):])

    def tensor(self, other: Layer) -> Layer:
        """ Tensor layers, coalescing their touching permutations. """
        assert_isinstance(other, type(self))
        *head, left = self
        right, *tail = other
        return type(self)(*head, left @ right, *tail)

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            return self.tensor(other)
        return super().__matmul__(other)

    def dagger(self) -> Layer:
        return type(self)(*(value.dagger() for value in self))

    @property
    def boxes_and_offsets(self) -> list[tuple[monoidal.Box, int]]:
        if any(not permutation.is_identity
               for permutation in self.permutations):
            raise NotImplementedError(messages.PERMUTATION_HAS_NO_OFFSET)
        if self.is_permutation:
            return []
        return super().boxes_and_offsets

    def merge(self, other: Layer) -> Layer:
        if any(not permutation.is_identity
               for layer in (self, other)
               for permutation in layer.permutations):
            raise AxiomError(messages.NOT_MERGEABLE.format(self, other))
        if self.is_permutation:
            assert_iscomposable(self, other)
            return other
        if other.is_permutation:
            assert_iscomposable(self, other)
            return self
        return super().merge(other)


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
    layer_factory = Layer
    twist_factory = classmethod(lambda cls, dom: cls.id(dom))

    def setoid(self):
        if len(self.inside) == 1\
                and isinstance(self.inside[0], Layer)\
                and self.inside[0].is_permutation:
            return self.inside[0].permutation.setoid()
        return super().setoid()

    @property
    def has_nonidentity_permutation(self) -> bool:
        """ Whether one of the layers has non-trivial structural routing. """
        return any(
            isinstance(layer, Layer)
            and any(not permutation.is_identity
                    for permutation in layer.permutations)
            for layer in self.inside)

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
        The diagram that encodes a given permutation as a composition of
        swaps.

        Parameters:
            xs : A permutation, as a sequence of integers or a
                 :class:`finset.Permutation`.
            dom : A type of the same length as :code:`xs`,
                  default is :code:`PRO(len(xs))`.
        """
        xs = list(xs)
        dom = PRO(len(xs)) if dom is None else dom
        if xs == list(range(len(xs))):
            return cls.id(dom)
        if list(range(len(dom))) != sorted(xs):
            raise ValueError(messages.WRONG_PERMUTATION.format(len(dom), xs))
        i = xs[0]
        return cls.swap(dom[:i], dom[i]) @ dom[i + 1:]\
            >> dom[i] @ cls.permutation(
                [x - 1 if x > i else x for x in xs[1:]], dom[:i] + dom[i + 1:])

    @classmethod
    def from_permutation(cls, perm: Sequence[int], dom: monoidal.Ty = None
                         ) -> Diagram:
        """
        Encode a permutation natively when the category has a matching
        :class:`Permutation` factory. Descendant categories without one use
        their own swap decomposition instead. An identity permutation always
        becomes the identity diagram.

        Parameters:
            perm : A permutation, as a sequence of integers or a
                   :class:`finset.Permutation`.
            dom : A type of the same length as :code:`perm`,
                  default is :code:`PRO(len(perm))`.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Diagram.from_permutation([1, 2, 0], x @ y @ z)\\
        ...     == Permutation(x @ y @ z, [1, 2, 0])
        >>> assert Diagram.from_permutation(
        ...     [0, 1, 2], x @ y @ z) == Id(x @ y @ z)
        """
        dom = PRO(len(perm)) if dom is None else dom
        perm = finset.Permutation(perm, len(dom))
        if perm.is_identity:
            return cls.id(dom)
        if cls.permutation_factory.ar is cls:
            return cls.permutation_factory(dom, perm)
        return cls.permutation(perm, dom)

    def permute(self, *xs: int) -> Diagram:
        """
        Post-compose with a permutation written as the historical swap
        decomposition. Use :meth:`from_permutation` to construct a native
        :class:`Permutation` box.

        Parameters:
            xs : A list of integers representing a permutation.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Id(x @ y @ z).permute(2, 0, 1).cod == z @ x @ y
        """
        return self >> self.permutation(list(xs), self.cod)

    def simplify(self):
        """ Simplify by translating back and forth to hypergraph. """
        return self.to_hypergraph().to_diagram()

    def to_hypergraph(self) -> Hypergraph:
        """
        Translate to a hypergraph without asking structural permutations for
        legacy box offsets.
        """
        if self.has_nonidentity_permutation:
            return hypergraph.Hypergraph[type(self).ar].from_diagram(self)
        return super().to_hypergraph()

    def foliation(self):
        """
        Merge independent generators while keeping native routing compact.

        >>> x, y = Ty('x'), Ty('y')
        >>> perm = Permutation(x @ y, [1, 0])
        >>> assert perm.foliation() == perm
        """
        if self.has_nonidentity_permutation:
            return self._merge_layers()
        return super().foliation()

    def interchange(self, i: int, j: int, left=False) -> Diagram:
        if self.has_nonidentity_permutation:
            raise NotImplementedError(messages.PERMUTATION_HAS_NO_OFFSET)
        return super().interchange(i, j, left=left)

    def normalize(self, left=False):
        if self.has_nonidentity_permutation:
            raise NotImplementedError(messages.PERMUTATION_HAS_NO_OFFSET)
        return super().normalize(left=left)

    def substitute(self, i: int, other: Diagram) -> Diagram:
        if self.has_nonidentity_permutation:
            raise NotImplementedError(messages.PERMUTATION_HAS_NO_OFFSET)
        return super().substitute(i, other)

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

    Permutations remain boxes at the public API boundary, but layers store
    them at even routing positions and exclude them from their generator list.
    Identity permutations are constructible as routing components and compare
    equal to the corresponding identity diagram.

    Parameters:
        dom : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.

    Examples
    --------
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> perm = Permutation(x @ y @ z, [1, 2, 0])
    >>> assert perm.cod == y @ z @ x
    >>> assert perm.dagger() == Permutation(y @ z @ x, [2, 0, 1])
    >>> assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    """
    def __init__(self, dom: monoidal.Ty, perm: Sequence[int]):
        self.perm = finset.Permutation(perm, len(dom))
        cod = dom[:0].tensor(*(dom[i] for i in self.perm))
        super().__init__(
            f"Permutation({list(self.perm)})", dom, cod,
            draw_as_wires=True, drawing_permutation=tuple(self.perm))
        if self.perm.is_identity:
            self.inside = ()

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.is_identity:
            self.inside = ()

    @property
    def is_identity(self) -> bool:
        """
        Whether the underlying permutation is the identity.

        >>> assert Permutation(Ty('x', 'y'), [0, 1]).is_identity
        """
        return self.perm.is_identity

    @property
    def size(self) -> int:
        """ Structural permutations are not generator boxes in a layer. """
        return 0

    def setoid(self):
        if self.is_identity:
            return (), self.dom, self.cod
        return type(self), self.dom, tuple(self.perm)

    def to_drawing(self):
        """ Draw as a compact band, or as wires for the identity. """
        from discopy.drawing import Drawing
        return Drawing.id(self.dom) if self.is_identity\
            else Drawing.from_box(self)

    def to_swaps(self) -> Diagram:
        """
        The same permutation built as a composition of swaps.

        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> perm = Permutation(x @ y @ z, [1, 2, 0])
        >>> assert Equation(perm.to_swaps(), perm)
        """
        return self.ar.permutation(self.perm, self.dom)

    def to_tree(self) -> dict:
        """
        Serialise a permutation, see :func:`discopy.utils.dumps`.

        >>> from discopy.utils import dumps, loads
        >>> x, y = Ty('x'), Ty('y')
        >>> assert loads(dumps(Permutation(x @ y, [1, 0])))\\
        ...     == Permutation(x @ y, [1, 0])
        """
        return dict(factory=factory_name(type(self)),
                    dom=self.dom.to_tree(), perm=list(self.perm))

    @classmethod
    def from_tree(cls, tree: dict) -> Permutation:
        return cls(from_tree(tree['dom']), tree['perm'])

    def dagger(self) -> Permutation:
        return type(self)(self.cod, self.perm.dagger())

    def tensor(self, other=None, *others):
        if other is None:
            return self
        if isinstance(other, Permutation):
            result = type(self)(
                self.dom @ other.dom, self.perm.tensor(other.perm))
        elif isinstance(other, monoidal.Ty)\
                or isinstance(other, Diagram) and not other.inside:
            typ = other if isinstance(other, monoidal.Ty) else other.dom
            result = type(self)(self.dom @ typ, self.perm.tensor(
                finset.Permutation.id(len(typ))))
        else:
            result = super().tensor(other)
        return result.tensor(*others)

    def __rmatmul__(self, other):
        if not isinstance(other, monoidal.Ty):
            return super().__rmatmul__(other)
        perm = finset.Permutation.id(len(other)).tensor(self.perm)
        return type(self)(other @ self.dom, perm)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.dom!r}, {list(self.perm)})"

    def __str__(self):
        return f"Permutation({self.dom}, {list(self.perm)})"


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
            return self.cod.ar.swap(self(other.dom[0]), self(other.dom[1]))
        if isinstance(other, Permutation):
            return self.cod.ar.permutation(other.perm, self(other.dom))
        return super().__call__(other)


class CMap(traced.CMap):
    category = Diagram
    require_planar = False


Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
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
