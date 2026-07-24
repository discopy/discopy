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

Foliation
=========

:meth:`Diagram.to_layers` foliates a diagram into the smallest list of
symmetric layers, scheduling boxes as early as possible and absorbing swaps
into native permutations. :meth:`Diagram.foliation_drawing` then draws this
foliation compactly: each permutation becomes a single band of crossing
wires rather than a staircase of swaps, so the quadratic number of swaps
collapses into a linear number of permutation layers.

>>> f0, f1 = Box("f0", w, x), Box("f1", z, y)
>>> g0, g1 = Box("g0", y, z), Box("g1", x, w)
>>> diagram = Id(x @ y @ z @ w).permute(3, 2, 1, 0) >> f0 @ f1 @ g0 @ g1
>>> diagram.depth()
1

The diagram reverses four wires before a single layer of boxes, so its
foliation has just one permutation layer and one box layer:

>>> from discopy.drawing import Equation
>>> Equation(diagram, diagram.foliation_drawing(), symbol="=").draw(
...     path='docs/_static/symmetric/foliation.png', figsize=(7, 6),
...     margins=(0.05, 0.03))

.. image:: /_static/symmetric/foliation.png
    :align: center
"""

from __future__ import annotations

from discopy import monoidal, balanced, traced, messages, hypergraph
from discopy.abc import SymmetricCategory
from discopy.cat import factory
from discopy.monoidal import Wire, Ty, PRO  # noqa: F401
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

    def to_drawing(self):
        """Draw as a single compact band of crossing wires."""
        from discopy.drawing import Drawing
        return Drawing.permutation(self.inside, self.dom)

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

    A layer with no boxes is just a single :class:`Permutation`, i.e.
    ``Layer(perm)``. This makes a symmetric diagram uniformly a list of
    layers, with permutation-only layers routing wires between box-layers.

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

    A permutation-only layer has length one:

    >>> perm_layer = Layer(Permutation([1, 0], x @ y))
    >>> assert perm_layer.dom == x @ y and perm_layer.cod == y @ x
    >>> assert perm_layer.boxes == ()
    """
    def __init__(self, *inside):
        if len(inside) < 1 or len(inside) % 2 == 0:
            raise ValueError(
                "Layer needs an odd number of elements (>= 1).")
        for i, x in enumerate(inside):
            if i % 2 == 0:
                assert_isinstance(x, Permutation)
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

    def to_drawing(self):
        """
        Draw the layer compactly, with each permutation rendered as a single
        band of crossing wires rather than a staircase of swaps.
        """
        result = None
        for x in self.inside:
            d = x.to_drawing()
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

    @classmethod
    def from_layers(cls, layers: list[Layer], dom: monoidal.Ty = None
                    ) -> Diagram:
        """
        Rebuild a diagram from a list of symmetric :class:`Layer`, i.e. the
        inverse of :meth:`to_layers`.

        Parameters:
            layers : The list of layers, composed sequentially.
            dom : The domain, needed only if ``layers`` is empty.

        Examples
        --------
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', y, x)
        >>> d = f >> g
        >>> assert Diagram.from_layers(d.to_layers()) == d
        """
        if not layers:
            return cls.id(cls.ty_factory() if dom is None else dom)
        result = cls.id(layers[0].dom)
        for layer in layers:
            result = result >> layer.to_diagram()
        return result

    def to_layers(self) -> list[Layer]:
        """
        Foliate into a list of symmetric :class:`Layer`, encoding permutations
        natively rather than as networks of swaps.

        At each step every box whose inputs are available happens, scanning the
        wires from left to right. A box whose inputs are already lined up in
        the right order happens in one step; a box whose inputs are all present
        but out of order is plugged in two steps, i.e. after a
        :class:`Permutation`-only layer that lines it up. A single permutation
        reorders all the boxes that need it at once and leaves the others in
        place, so a permutation only appears when some box genuinely needs one.
        This keeps the box-layer count equal to the depth while avoiding the
        gratuitous swaps of an ASAP schedule that gathers every box to the
        left. Identity permutations are dropped and consecutive
        permutation-only layers are merged.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)

        Sequential boxes give one box-layer each:

        >>> layers = (f >> g).to_layers()
        >>> assert [bool(l.boxes) for l in layers] == [True, True]

        Parallel boxes already in order happen together, with no permutation:

        >>> layers = (f @ g.dom >> f.cod @ g).to_layers()
        >>> assert len(layers) == 1 and len(layers[0].boxes) == 2

        Swaps become permutation-only layers and round-trip on the nose:

        >>> d = f @ Box('h', z, x) >> Swap(y, x)
        >>> with Diagram.hypergraph_equality:
        ...     assert Diagram.from_layers(d.to_layers()) == d
        """
        if any(isinstance(box, Sum) for box in self.boxes):
            raise NotImplementedError
        empty = self.dom[:0]
        n = len(self.dom)
        # --- Pass 1: track wires, recording each box's input and output wires.
        frontier = list(range(n))
        wire_type = {i: self.dom[i] for i in range(n)}
        next_id, records = n, []  # records: (box, in_wids, out_wids)
        for layer in self.inside:
            delta = 0
            for box, offset in layer.boxes_and_offsets:
                off = offset + delta
                dim_dom, dim_cod = len(box.dom), len(box.cod)
                in_wids = frontier[off:off + dim_dom]
                if isinstance(box, Swap):  # A swap just permutes two wires.
                    frontier[off:off + dim_dom] = in_wids[::-1]
                    continue
                out_wids = list(range(next_id, next_id + dim_cod))
                next_id += dim_cod
                for i, w in enumerate(out_wids):
                    wire_type[w] = box.cod[i]
                frontier[off:off + dim_dom] = out_wids
                delta += dim_cod - dim_dom
                records.append((box, tuple(in_wids), tuple(out_wids)))
        output_wids = list(frontier)

        def type_of(wids):
            return empty.tensor(*[wire_type[w] for w in wids])

        def route(source, target):
            index = {w: i for i, w in enumerate(source)}
            return Layer(Permutation(
                [index[w] for w in target], type_of(source)))

        # --- Pass 2: at each step, fire every box whose inputs are available,
        # scanning left to right. A box whose inputs are already lined up
        # happens in one step; one whose inputs are all present but out of
        # order is plugged in two steps, i.e. after a permutation that lines it
        # up. A single permutation handles all the boxes that need reordering
        # at once and leaves the others in place, so a permutation only appears
        # when some box genuinely needs it.
        result, frontier, remaining = [], list(range(n)), list(records)
        while remaining:
            available = [r for r in remaining if set(r[1]) <= set(frontier)]
            by_wire = {w: r for r in available for w in r[1]}
            # Empty-domain boxes have no trigger wire, so happen straightaway.
            layout = [("box", r) for r in available if not r[1]]
            target, placed = [], set()
            for w in frontier:
                if w in placed:
                    continue
                rec = by_wire.get(w)
                if rec is None:
                    layout.append(("wire", w))
                    target.append(w)
                    placed.add(w)
                else:  # Line up this box's whole input block in order.
                    layout.append(("box", rec))
                    target += list(rec[1])
                    placed.update(rec[1])
            if target != frontier:
                result.append(route(frontier, target))
            inside, block = [], []
            for kind, val in layout:
                if kind == "wire":
                    block.append(val)
                else:
                    inside += [Permutation.id(type_of(block)), val[0]]
                    block = []
            inside.append(Permutation.id(type_of(block)))
            result.append(Layer(*inside))
            frontier = [w for kind, val in layout for w in (
                [val] if kind == "wire" else val[2])]
            done = set(map(id, available))
            remaining = [r for r in remaining if id(r) not in done]
        result.append(route(frontier, output_wids))

        # --- Pass 3: drop identities and merge consecutive permutations.
        layers = []
        for layer in result:
            if not layer.boxes and layer.inside[0].is_identity:
                continue
            if layers and not layers[-1].boxes and not layer.boxes:
                merged = layers[-1].inside[0].then(layer.inside[0])
                layers[-1] = Layer(merged)
                continue
            layers.append(layer)
        return layers or [Layer(Permutation.id(self.dom))]

    def foliation_drawing(self):
        """
        A compact :class:`Drawing` of the diagram's foliation.

        The diagram is foliated with :meth:`to_layers`, then each layer is
        drawn natively: box-layers as boxes side by side and permutation-only
        layers as a single band of crossing wires. The result has one row per
        layer, so it is much shorter than expanding every permutation into a
        staircase of swaps.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> p = Box('p', x, y @ z)
        >>> q, r, s = Box('q', y, x), Box('r', z, x), Box('s', x @ x, x)
        >>> d = p >> Swap(y, z) >> r @ q >> s
        >>> drawing = d.foliation_drawing()
        >>> assert drawing.dom == d.dom and drawing.cod == d.cod
        """
        from discopy.drawing import Drawing
        result = Drawing.id(self.dom)
        for layer in self.to_layers():
            result = result >> layer.to_drawing()
        return result

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


class CMap(traced.CMap):
    category = Diagram
    require_planar = False


Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.braid_factory = Swap
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
