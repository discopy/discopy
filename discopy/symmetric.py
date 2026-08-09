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
...     space=2, doctest='docs/_static/symmetric/hexagons.svg', figsize=(5, 2))

.. image:: /_static/symmetric/hexagons.svg
    :align: center

Involution
==========
a.k.a. Reidemeister move 2

>>> assert Swap(x, y)[::-1] == Swap(y, x)
>>> assert Equation(Swap(x, y) >> Swap(y, x), Id(x @ y))
>>> Equation(Swap(x, y) >> Swap(y, x), Id(x @ y)).draw(
...     doctest='docs/_static/symmetric/inverse.svg', figsize=(3, 2))

.. image:: /_static/symmetric/inverse.svg
    :align: center

Naturality
==========

>>> naturality = Equation(
...     f @ g >> Swap(f.cod, g.cod), Swap(f.dom, g.dom) >> g @ f)
>>> assert naturality
>>> naturality.draw(
...     doctest='docs/_static/symmetric/naturality.svg', figsize=(3, 2))

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
...     doctest='docs/_static/symmetric/yang-baxter.svg', figsize=(3, 2))

.. image:: /_static/symmetric/yang-baxter.svg
    :align: center

"""

from __future__ import annotations

from collections.abc import Sequence

from discopy import monoidal, balanced, messages, hypergraph, cmap
from discopy.abc import SymmetricCategory
from discopy.cat import factory
from discopy.monoidal import Wire, Ty, PRO  # noqa: F401
from discopy.python import finset
from discopy.utils import (
    assert_isinstance, classproperty, factory_name,
    from_tree)


class Layer(monoidal.Layer):
    """
    A tensor product :math:`s_0 \\otimes f_1 \\otimes \\dots \\otimes f_n
    \\otimes s_n` of generators :math:`f_i` and routing :math:`s_i`, where
    routing is a type when it is the identity and a :class:`Permutation`
    otherwise. :class:`Swap` is a generator, distinct from ``[1, 0]``.

    Routing components are coalesced, so a permutation given between two
    types becomes one permutation and the layer stays alternating. An
    identity permutation is stored as its type, hence a layer with a single
    permutation always permutes: the identity is the empty diagram, not a
    layer.

    A layer with no crossing is stored exactly as a
    :class:`discopy.monoidal.Layer`.

    Parameters:
        inside : Alternating routing and generators, starting and ending
                 with routing.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, perm = Box('f', x, y), Permutation(x @ y, [1, 0])
    >>> assert Layer(x, f, y).boxes_or_types == (x, f, y)
    >>> assert Layer(x, f, perm).boxes_or_types == (x, f, perm)
    >>> assert Layer(x, perm, y) == Layer(Permutation(x @ x @ y @ y,
    ...     [0, 2, 1, 3]))

    Forgetting the distinction between routing and generators gives the
    ordinary alternating view of a :class:`discopy.monoidal.Layer`, which is
    what :attr:`boxes`, :attr:`boxes_and_offsets` and the rewrites indexed by
    them are computed from.

    >>> assert Layer(x, f, perm).boxes_and_types == (
    ...     x, f, Ty(), perm, Ty())
    >>> assert Layer(x, f, perm).boxes_and_offsets == [(f, 1), (perm, 2)]
    """
    def __init__(self, *inside):
        boxes_or_types = []
        for value in inside:
            if boxes_or_types and self.is_routing(value)\
                    and self.is_routing(boxes_or_types[-1]):
                boxes_or_types[-1] = boxes_or_types[-1] @ value
            else:
                boxes_or_types.append(value)
        if not len(boxes_or_types) % 2 or any(
                self.is_routing(value) != (not i % 2)
                for i, value in enumerate(boxes_or_types)):
            raise ValueError(messages.LAYERS_MUST_ALTERNATE)
        super().__init__(*(
            value.dom if isinstance(value, Permutation) and value.is_identity
            else value for value in boxes_or_types))

    @staticmethod
    def is_routing(value) -> bool:
        """
        Whether a component routes wires rather than generating them.

        >>> x = Ty('x')
        >>> assert Layer.is_routing(x) and Layer.is_routing(
        ...     Permutation(x @ x, [1, 0]))
        >>> assert not Layer.is_routing(Box('f', x, x))
        """
        return isinstance(value, (monoidal.Ty, Permutation))

    @property
    def is_structural(self) -> bool:
        """
        Whether the layer routes its wires non-trivially, i.e. one of its
        routing components is a :class:`Permutation` rather than a type.

        >>> x, y = Ty('x'), Ty('y')
        >>> assert Layer(Permutation(x @ y, [1, 0])).is_structural
        >>> assert not Layer(x, Box('f', x, y), y).is_structural
        """
        return any(isinstance(value, Permutation) for value in self)

    @property
    def boxes_and_types(self):
        """ Every permutation as an ordinary box between empty types. """
        result = []
        for i, value in enumerate(self.boxes_or_types):
            if i % 2 or not isinstance(value, Permutation):
                result.append(value)
            else:
                result += [
                    value.dom[:0], value, value.cod[len(value.cod):]]
        return tuple(result)

    @classmethod
    def cast(cls, box: Box) -> Layer:
        """ Turn a generator or a permutation into a layer. """
        if isinstance(box, Permutation):
            return cls(box)
        return cls(box.dom[:0], box, box.cod[len(box.cod):])

    def tensor(self, other: Layer) -> Layer:
        """ Tensor layers, coalescing their touching routing. """
        assert_isinstance(other, type(self))
        return type(self)(*self, *other)

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            return self.tensor(other)
        return super().__matmul__(other)


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
    ...              doctest='docs/_static/symmetric/decorator.svg')

    .. image:: /_static/symmetric/decorator.svg
        :align: center

    Every variable must be used exactly once or this will raise an error.

    >>> from pytest import raises
    >>> from discopy.utils import AxiomError

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, x @ x)(lambda x: (x, x))
    >>> print(err.value)
    symmetric.Diagram has no spiders, cups or caps to downgrade this \
hypergraph.

    >>> with raises(AxiomError) as err:
    ...     Diagram.from_callable(x, Ty())(lambda x: ())
    >>> print(err.value)
    symmetric.Diagram has no spiders, cups or caps to downgrade this \
hypergraph.

    Note
    ----
    As for :class:`discopy.balanced.Diagram`, our symmetric diagrams are traced
    by default. However now we have that the axioms for trace hold on the nose.

    Note
    ----
    The swaps of atomic types are generated by :attr:`swap_factory`, which
    subclasses should set to their own subclass of :class:`Swap`. It is the
    braid of a symmetric category, i.e. :attr:`braided.Diagram.braid_factory`
    reads it:

    >>> class Permutation(Diagram): ...
    >>> class Transposition(Swap, Permutation): ...
    >>> Permutation.swap_factory = Transposition
    >>> assert Permutation.braid_factory is Transposition
    """
    braid_factory = classproperty(lambda cls: cls.swap_factory)
    layer_factory = Layer
    twist_factory = classmethod(lambda cls, dom: cls.id(dom))

    @property
    def is_structural(self) -> bool:
        """ Whether one of the layers routes its wires non-trivially. """
        return any(layer.is_structural for layer in self.inside)

    @classmethod
    def swap(cls, left: monoidal.Ty, right: monoidal.Ty) -> Diagram:
        """
        The diagram that swaps the ``left`` and ``right`` wires.

        Parameters:
            left : The type at the top left and bottom right.
            right : The type at the top right and bottom left.

        Note
        ----
        This calls :func:`balanced.hexagon` and :attr:`swap_factory`.
        """
        return cls.braid(left, right)

    @classmethod
    def permutation(cls, xs: Sequence[int],
                    doms: Sequence[monoidal.Ty] | None = None) -> Diagram:
        """
        The diagram that encodes a given permutation as a composition of
        swaps.

        Parameters:
            xs : A permutation, as a sequence of integers or a
                 :class:`finset.Permutation`.
            dom : A type of the same length as :code:`xs`,
                  default is :code:`PRO(len(xs))`.
        """

        doms = PRO(len(xs)) if doms is None else doms
        size = len(doms)
        unit = type(doms)() if isinstance(doms, PRO) else cls.ob()
        tensor = lambda tys: unit.tensor(*tys)
        dom = tensor(doms)

        xs = finset.Permutation(xs, size)
        if xs.is_identity:
            return cls.id(dom)
        i = xs[0]
        left, head, right = (
            doms[slice]
            for slice in (
                slice(0, i), i, slice(i + 1, None)
            )
        )
        return cls.swap(tensor(left), head) @ tensor(right)\
            >> head @ cls.permutation(
                [x - 1 if x > i else x for x in xs[1:]],
                left + right)

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

    def foliation(self):
        """
        Merge independent generators, keeping native routing compact.

        A hypergraph forgets that routing is native, so a diagram with a
        :class:`Permutation` is foliated by merging its layers instead.

        >>> x, y = Ty('x'), Ty('y')
        >>> perm = Permutation(x @ y, [1, 0])
        >>> assert perm.foliation() == perm
        """
        if self.is_structural:
            return self.merge_layers()
        return super().foliation()

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

    A :class:`Layer` stores it as routing rather than as a generator, and the
    identity permutation is the identity diagram. It draws as a single band of
    crossing wires rather than a staircase of swaps.

    Parameters:
        dom : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.

    Examples
    --------
    >>> x, y, z, w = map(Ty, "xyzw")
    >>> perm = Permutation(x @ y @ z, [1, 2, 0])
    >>> assert perm.cod == y @ z @ x
    >>> assert perm.dagger() == Permutation(y @ z @ x, [2, 0, 1])
    >>> assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    >>> assert perm @ Id(w) == Permutation(x @ y @ z @ w, [1, 2, 0, 3])
    >>> assert Permutation(x @ y, [1, 0]) != Swap(x, y)
    >>> assert Equation(Permutation(x @ y, [1, 0]), Swap(x, y))
    >>> assert Permutation(x @ y, [0, 1]) == Id(x @ y)

    Writing permutations by hand keeps swap-heavy diagrams compact: a whole
    permutation occupies a single layer rather than a quadratic staircase of
    swaps. Reversing four wires before a single layer of boxes is a
    permutation layer followed by a box layer.

    >>> f0, f1 = Box("f0", w, x), Box("f1", z, y)
    >>> g0, g1 = Box("g0", y, z), Box("g1", x, w)
    >>> reverse = Permutation(x @ y @ z @ w, [3, 2, 1, 0])
    >>> diagram = reverse >> f0 @ f1 @ g0 @ g1
    >>> diagram.depth()
    1
    >>> diagram.draw(
    ...     doctest='docs/_static/symmetric/foliation.svg', figsize=(4, 4))

    .. image:: /_static/symmetric/foliation.svg
        :align: center
    """

    def __init__(self, dom: monoidal.Ty, perm: Sequence[int]):
        self.perm = finset.Permutation(perm, len(dom))
        cod = dom[:0].tensor(*(dom[i] for i in self.perm))
        name = f"Permutation({list(self.perm)})"
        super().__init__(
            name, dom, cod, drawing_name=name,
            draw_as_wires=True, draw_as_permutation=True,
            permutation_indices=tuple(self.perm))

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
        doms = self.dom if isinstance(self.dom, PRO)\
            else list(map(self.ob, self.dom.inside))
        return self.ar.permutation(self.perm, doms)

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
            if isinstance(other.dom, PRO):
                doms = self(other.dom)
            else:
                doms = list(map(self, other.dom))
            return self.cod.ar.permutation(other.perm, doms)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.swap_factory = Swap
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
