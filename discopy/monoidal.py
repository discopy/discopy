# -*- coding: utf-8 -*-

"""
The free (pre)monoidal category, i.e. planar diagrams.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    PRO
    Layer
    Diagram
    Box
    Sum
    Bubble
    Category
    Functor
    Whiskerable

Axioms
------
We can check the axioms for :class:`Ty` being a monoid.

>>> x, y, z, unit = Ty('x'), Ty('y'), Ty('z'), Ty()
>>> assert x @ unit == x == unit @ x
>>> assert (x @ y) @ z == x @ y @ z == x @ (y @ z)

We can check the axioms for dagger monoidal categories, up to interchanger.

>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
>>> d = Id(x) @ f1 >> f0 @ Id(w)
>>> assert d == (f0 @ f1).interchange(0, 1)
>>> assert f0 @ f1 == d.interchange(0, 1)
>>> assert (f0 @ f1)[::-1][::-1] == f0 @ f1
>>> assert (f0 @ f1)[::-1].interchange(0, 1) == f0[::-1] @ f1[::-1]

We can check the Eckmann-Hilton argument, up to interchanger.

>>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
>>> assert s0 @ s1 == s0 >> s1 == (s1 @ s0).interchange(0, 1)
>>> assert s1 @ s0 == s1 >> s0 == (s0 @ s1).interchange(0, 1)

.. image:: /_static/monoidal/EckmannHilton.gif
    :align: center
"""

from __future__ import annotations

import itertools
from typing import Iterator, Callable, TYPE_CHECKING
from dataclasses import dataclass
from warnings import warn

from discopy import cat, drawing, hypergraph, messages
from discopy.cat import Ob
from discopy.utils import (
    factory,
    factory_name,
    from_tree,
    assert_isinstance,
    assert_iscomposable,
    Whiskerable,
    AxiomError,
)

if TYPE_CHECKING:
    import sympy


@factory
class Ty(cat.Ob):
    """
    A type is a tuple of objects with :meth:`Ty.tensor` as concatenation.

    Parameters:
        inside : The objects inside the type (or their names).

    Tip
    ---
    Types can be instantiated with a name rather than object.

    >>> assert Ty('x') == Ty(cat.Ob('x'))

    Tip
    ---
    A type can be exponentiated by a natural number.

    >>> assert Ty('x') ** 3 == Ty('x', 'x', 'x')

    Note
    ----
    Types can be indexed and sliced using square brackets. Indexing behaves
    like that of strings, i.e. when we index a type we get a type back.
    The objects inside the type are still accessible using ``.inside``.

    >>> t = Ty(*"xyz")
    >>> assert t[0] == t[:1] == Ty('x')
    >>> assert t[0] != t.inside[0] == Ob('x')
    >>> assert t[1:] == t[-2:] == Ty('y', 'z')
    """
    ob_factory = cat.Ob

    __ambiguous_inheritance__ = True

    def __setstate__(self, state):
        if 'inside' not in state and "_objects" in state:
            state["inside"] = state['_objects']
            del state['_objects']
        super().__setstate__(state)

    def __init__(self, *inside: str | cat.Ob):
        for obj in inside:
            assert_isinstance(obj, (str, self.ob_factory))
        self.inside = tuple(x if isinstance(x, self.ob_factory)
                            else self.ob_factory(x) for x in inside)
        super().__init__(str(self))

    def tensor(self, *others: Ty) -> Ty:
        """
        Returns the tensor of types, i.e. the concatenation of their lists
        of objects. This is called with the binary operator :code:`@`.

        Parameters:
            others : The other types to tensor.

        Tip
        ---
        A list of types can be tensored by calling :code:`Ty().tensor`.

        >>> list_of_types = [Ty('x'), Ty('y'), Ty('z')]
        >>> assert Ty().tensor(*list_of_types) == Ty('x', 'y', 'z')
        """
        for other in others:
            if not isinstance(other, Ty):
                return NotImplemented
            assert_isinstance(self, other.factory)
            assert_isinstance(other, self.factory)
        inside = self.inside + tuple(x for t in others for x in t.inside)
        return self.factory(*inside)

    def count(self, obj: cat.Ob) -> int:
        """
        Counts the occurrence of a given object (or a type of length 1).

        Parameters:
            obj : The object to count.

        Example
        -------

        >>> x = Ty('x')
        >>> xs = x ** 5
        >>> assert xs.count(x) == xs.inside.count(x.inside[0])
        """
        obj, = obj.inside if isinstance(obj, Ty) else (obj, )
        return self.inside.count(obj)

    def to_drawing(self) -> Ty:
        """ Called before :meth:`Diagram.draw`. """
        def obj_to_drawing(obj):
            result = cat.Ob(str(obj))
            result.always_draw_label = getattr(obj, "always_draw_label", False)
            return result
        return Ty(*map(obj_to_drawing, self.inside))

    @property
    def is_atomic(self) -> bool:
        """ Whether a type is atomic, i.e. it has length 1. """
        return len(self) == 1

    def __eq__(self, other):
        return isinstance(other, self.factory) and self.inside == other.inside

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.inside))})"

    def __str__(self):
        return ' @ '.join(map(str, self.inside)) or type(self).__name__ + '()'

    def __len__(self):
        return len(self.inside)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.factory(*self.inside[key])
        return cat.Arrow.__getitem__(self, key)

    def __pow__(self, n_times):
        return self.factory().tensor(*n_times * [self])

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'inside': [x.to_tree() for x in self.inside]}

    @classmethod
    def from_tree(cls, tree):
        if "inside" not in tree:
            warn("Outdated dumps", DeprecationWarning)
            return cls(*map(from_tree, tree['objects']))
        return cls(*map(from_tree, tree['inside']))

    def __matmul__(self, other):
        return self.tensor(other)

    __add__ = __matmul__


@factory
class PRO(Ty):
    """
    A PRO is a natural number ``n`` seen as a type with addition as tensor.

    Parameters
    ----------
    n : int
        The length of the PRO type.

    Example
    -------
    >>> assert PRO(1) @ PRO(2) == PRO(3)

    Note
    ----
    If ``ty_factory`` is ``PRO`` then :class:`Diagram` will automatically turn
    any ``n: int`` into ``PRO(n)``. Thus ``PRO`` never needs to be called.

    >>> @factory
    ... class Circuit(Diagram):
    ...     ty_factory = PRO
    >>> class Gate(Box, Circuit): ...
    >>> CX = Gate('CX', 2, 2)

    >>> assert CX @ 2 >> 2 @ CX == CX @ CX
    """
    def __init__(self, n: int = 0):
        assert_isinstance(n, int)
        self.n = n

    def __setstate__(self, state):
        if "n" not in state:
            state = {"n": len(state["_objects"])}
        super().__setstate__(state)

    @property
    def inside(self):
        return self.n * (1, )

    def tensor(self, *others: PRO) -> PRO:
        for other in others:
            if not isinstance(other, Ty):
                return NotImplemented  # This allows whiskering on the left.
            assert_isinstance(self, other.factory)
            assert_isinstance(other, self.factory)
        return self.factory(self.n + sum(other.n for other in others))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.factory(len(self.inside[key]))
        return cat.Arrow.__getitem__(self, key)

    def __len__(self):
        return self.n

    def __repr__(self):
        return factory_name(type(self)) + f"({self.n})"

    def __str__(self):
        return f"PRO({self.n})"

    def __eq__(self, other):
        return isinstance(other, self.factory) and self.n == other.n

    def __hash__(self):
        return hash(repr(self))

    def __pow__(self, n_times):
        return self.factory(n_times * self.n)

    def to_drawing(self):
        return Ty(*self.n * [Ob()])

    def to_tree(self):
        return {'factory': factory_name(type(self)), 'n': self.n}

    @classmethod
    def from_tree(cls, tree):
        return cls(tree['n'])


class Layer(cat.Box):
    """
    A layer is a :code:`box` in the middle of a pair of types
    :code:`left` and :code:`right`.

    Parameters:
        left : The type on the left of the layer.
        box : The box in the middle of the layer.
        right : The type on the right of the layer.
        more : More boxes and types to the right,
               used by :meth:`Diagram.foliation`.
    """
    def __setstate__(self, state):
        if 'boxes_or_types' not in state:  # Backward compatibility
            self.boxes_or_types = tuple(
                state[key] for key in ['_left', '_box', '_right'])
            del state['_left'], state['_box'], state['_right']
        super().__setstate__(state)

    def __init__(self, left: Ty, box: Box, right: Ty, *more):
        if len(more) % 2:
            raise ValueError(messages.LAYERS_MUST_BE_ODD)
        self.boxes_or_types = (left, box, right) + more
        name, dom, cod = "", left[:0], left[:0]
        for i, box_or_typ in enumerate(self.boxes_or_types):
            if i % 2:
                assert_isinstance(box, Box)
                dom, cod = dom @ box_or_typ.dom, cod @ box_or_typ.cod
                name += ("" if not name else " @ ") + str(box_or_typ)
            else:
                assert_isinstance(box_or_typ, Ty)
                dom, cod = dom @ box_or_typ, cod @ box_or_typ
                name += "" if not box_or_typ\
                    else ("" if not name else " @ ") + str(box_or_typ)
        super().__init__(name, dom, cod)

    def __iter__(self):
        for box_or_typ in self.boxes_or_types:
            yield box_or_typ

    def __getitem__(self, key):
        return self.boxes_or_types[key]

    def __eq__(self, other):
        return isinstance(other, type(self)) and tuple(self) == tuple(other)

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self))})"

    def __matmul__(self, other: Ty) -> Layer:
        *tail, head = self
        return type(self)(*tail + [head @ other])

    def __rmatmul__(self, other: Ty) -> Layer:
        head, *tail = self
        return type(self)(other @ head, *tail)

    @property
    def free_symbols(self) -> "set[sympy.Symbol]":
        return {x for _, box, _ in self.inside for x in box.free_symbols}

    def subs(self, *args) -> Layer:
        left, box, right = self
        return type(self)(left, box.subs(*args), right)

    @classmethod
    def cast(cls, box: Box) -> Layer:
        """
        Turns a box into a layer with empty types on the left and right.

        Parameters:
            box : The box in the middle of empty types.

        Example
        -------
        >>> f = Box('f', Ty('x'), Ty('y'))
        >>> assert Layer.cast(f) == Layer(Ty(), f, Ty())
        """
        return cls(box.dom[:0], box, box.cod[len(box.cod):])

    def dagger(self) -> Layer:
        return type(self)(*(
            x.dagger() if i % 2 else x for i, x in enumerate(self)))

    def to_drawing(self) -> Diagram:
        """ Called before :meth:`Diagram.draw`. """
        result = Ty()
        for box_or_typ in self:
            result = result @ box_or_typ.to_drawing()
        return result

    @property
    def boxes_and_offsets(self) -> list[tuple[Box, int]]:
        """
        The offsets of each box inside the layer.

        Example
        -------
        >>> a, b, c, d, e = map(Ty, "abcde")
        >>> f, g = Box('f', a, b), Box('g', c, d)
        >>> assert Layer(e, f, e, g, e).boxes_and_offsets == [(f, 1), (g, 3)]
        """
        left, box, *tail = self
        boxes, offsets = [box], [len(left)]
        for typ, box in zip(tail[::2], tail[1::2]):
            boxes.append(box)
            offsets.append(offsets[-1] + len(boxes[-1].dom) + len(typ))
        return list(zip(boxes, offsets))

    def merge(self, other: Layer) -> Layer:
        """
        Merge two layers into one or raise :class:`AxiomError`,
        used by :meth:`Diagram.foliation`.

        Parameters:
            other : The other layer with which to merge.

        Example
        -------
        >>> a, b, c, d, e = map(Ty, "abcde")
        >>> f, g = Box('f', a, b), Box('g', c, d)
        >>> layer0 = Layer(e,  f,  e @ c @ e)
        >>> layer1 = Layer(e @ b @ e,  g,  e)
        >>> assert layer0.merge(layer1) == Layer(e, f, e, g, e)
        """
        assert_iscomposable(self, other)
        try:
            diagram = Diagram.normal_form(self.boxes_or_types[1].factory(
                (self, other), self.dom, other.cod).to_staircases())
        except NotImplementedError as exception:  # Eckmann-Hilton argument.
            diagram = exception.last_step
        boxes_or_types, offset = [self.dom[:0]], 0
        for layer in diagram.inside:
            left, box, right = layer
            if len(left) < offset:
                raise AxiomError(
                    messages.NOT_MERGEABLE.format(self, other))
            boxes_or_types[-1] @= left[offset:]
            boxes_or_types += [box, right[:0]]
            offset = len(left @ box.cod)
        boxes_or_types[-1] @= layer.cod[offset:]
        return type(self)(*boxes_or_types)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(*(
            x if not i % 2 else x.lambdify(*symbols, **kwargs)(*xs)
            for i, x in enumerate(self)))

    def to_tree(self) -> dict:
        return dict(factory=factory_name(type(self)),
                    inside=[x.to_tree() for x in self])

    @classmethod
    def from_tree(cls, tree: dict) -> Layer:
        return cls(*(map(from_tree, tree['inside'])))


@factory
class Diagram(cat.Arrow, Whiskerable):
    """
    A diagram is a tuple of composable layers :code:`inside` with a pair of
    types :code:`dom` and :code:`cod` as domain and codomain.

    Parameters:
        inside : The layers of the diagram.
        dom : The domain of the diagram, i.e. its input.
        cod : The codomain of the diagram, i.e. its output.

    .. admonition:: Summary

        .. autosummary::

            tensor
            boxes
            offsets
            draw
            interchange
            normalize
            normal_form
    """
    ty_factory = Ty
    layer_factory = Layer

    def __setstate__(self, state):
        if 'inside' not in state:  # Backward compatibility
            state |= {
                'dom': state['_dom'], 'cod': state['_cod'],
                'inside': tuple(state['_layers'])}
        super().__setstate__(state)

    def __init__(
            self, inside: tuple[Layer, ...], dom: Ty, cod: Ty, _scan=True):
        for layer in inside:
            assert_isinstance(layer, Layer)
        super().__init__(inside, dom, cod, _scan=_scan)

    @classmethod
    def from_callable(cls, dom: Ty, cod: Ty) -> Callable[Callable, Diagram]:
        """
        Define a diagram using the standard syntax for Python functions.

        Note that we can specify the offset as argument.

        Example
        -------
        >>> x = Ty('x')
        >>> cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
        >>> @Diagram.from_callable(x, x)
        ... def snake(left):
        ...     middle, right = cap(offset=1)
        ...     cup(left, middle)
        ...     return right
        >>> snake.draw(
        ...     figsize=(3, 3), path='docs/_static/drawing/diagramize.png')

        .. image:: /_static/drawing/diagramize.png
            :align: center
        """
        def decorator(func):
            hypergraph = cls.hypergraph_factory.from_callable(dom, cod)(func)
            return hypergraph.to_diagram()

        return decorator

    def tensor(self, other: Diagram = None, *others: Diagram) -> Diagram:
        """
        Parallel composition, called using :code:`@`.

        Parameters:
            other : The other diagram to tensor.
            rest : More diagrams to tensor.

        Important
        ---------
        The definition of tensor is biased to the left, i.e.::

            self @ other == self @ other.dom >> self.cod @ other

        Example
        -------
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> assert f0 @ f1 == f0.tensor(f1) == f0 @ Id(z) >> Id(y) @ f1

        >>> (f0 @ f1).draw(
        ...     figsize=(2, 2),
        ...     path='docs/_static/monoidal/tensor-example.png')

        .. image:: /_static/monoidal/tensor-example.png
            :align: center
        """
        if other is None:
            return self
        if others:
            return self.tensor(other).tensor(*others)
        if isinstance(other, Sum):
            return self.sum_factory((self, )).tensor(other)
        assert_isinstance(other, self.factory)
        assert_isinstance(self, other.factory)
        inside = tuple(layer @ other.dom for layer in self.inside)\
            + tuple(self.cod @ layer for layer in other.inside)
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        return self.factory(inside, dom, cod, _scan=False)

    @property
    def boxes(self) -> list[Box]:
        """ The boxes in each layer of the diagram. """
        return list(box for _, box, _ in self)

    @property
    def offsets(self) -> list[int]:
        """ The offset of a box is the length of the type on its left. """
        return list(len(left) for left, _, _ in self)

    @property
    def width(self):
        """
        The width of a diagram, i.e. the maximum number of parallel wires.

        Example
        -------
        >>> x = Ty('x')
        >>> f = Box('f', x, x ** 4)
        >>> diagram = f @ x ** 2 >> x ** 2 @ f.dagger()
        >>> assert diagram.width == 6
        """
        return max(len(self.dom), max(len(layer.cod) for layer in self))

    def encode(self) -> tuple[Ty, list[tuple[Box, int]]]:
        """
        Compact encoding of a diagram as a tuple of boxes and offsets.

        Example
        -------
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1, g = Box('f0', x, y), Box('f1', z, w), Box('g', y @ w, y)
        >>> diagram = f0 @ f1 >> g
        >>> dom, boxes_and_offsets = diagram.encode()
        >>> assert dom == x @ z
        >>> assert boxes_and_offsets == [(f0, 0), (f1, 1), (g, 0)]
        >>> assert diagram == Diagram.decode(*diagram.encode())
        >>> diagram.draw(figsize=(2, 2),
        ...        path='docs/_static/monoidal/arrow-example.png')

        .. image:: /_static/monoidal/arrow-example.png
            :align: center
        """
        return self.dom, list(zip(self.boxes, self.offsets))

    @classmethod
    def decode(
            cls,
            dom: Ty,
            boxes_and_offsets: list[tuple[Box, int]] = None,
            boxes: list[Box] = None,
            offsets: list[int] = None,
            cod: Ty = None) -> Diagram:
        """
        Turn a tuple of boxes and offsets into a diagram.

        Parameters:
            dom : The domain of the diagram.
            cod : The codomain of the diagram.
            boxes_and_offsets : The boxes and offsets of the diagram.
            boxes : The list of boxes.
            offsets : The list of offsets.

        Example
        -------
        >>> x, y, z, w = map(Ty, "xyzw")
        >>> f, g = Box('f', x, y), Box('g', z, w)
        >>> assert f @ z >> y @ g == Diagram.decode(
        ...     dom=x @ z, cod=y @ w, boxes=[f, g], offsets=[0, 1])

        Note
        ----
        If ``boxes_and_offsets is None``
        then we set it to ``zip(boxes, offstes)``.
        """
        if boxes_and_offsets is None:
            boxes_and_offsets = zip(boxes, offsets)
        diagram = cls.id(dom)
        for box, offset in boxes_and_offsets:
            left = diagram.cod[:offset]
            right = diagram.cod[offset + len(box.dom):]
            diagram = diagram >> left @ box @ right
        if cod is not None:
            assert_iscomposable(diagram, cls.id(cod))
        return diagram

    def to_drawing(self):
        """ Called before :meth:`Diagram.draw`. """
        return cat.Functor(
            ob=lambda x: x.to_drawing(), ar=Layer.to_drawing, cod=Category())(
                self)

    def to_staircases(self):  # pylint:
        """
        Splits layers with more than one box into staircases.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> diagram = y @ f0 >> f1 @ y
        >>> print(diagram.foliation())
        f1 @ f0
        >>> print(diagram.foliation().to_staircases())
        f1 @ x >> x @ f0
        """
        return Functor.id(Category(self.ty_factory, self.factory))(self)

    def foliation(self):
        """
        Merges layers together to reduce the length of a diagram.

        Example
        -------
        >>> from discopy.monoidal import *
        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> diagram = f0 @ Id(y) >> f0.dagger() @ f1
        >>> print(diagram)
        f0 @ y >> f0[::-1] @ y >> x @ f1
        >>> print(diagram.foliation())
        f0 @ y >> f0[::-1] @ f1

        Note
        ----
        If one defines a foliation as a sequence of unmergeable layers, there
        may exist many distinct foliations for the same diagram. This method
        scans top to bottom and merges layers eagerly.
        """
        while len(self) > 1:
            keep_on_going = False
            for i, (first, second) in enumerate(zip(
                    self.inside, self.inside[1:])):
                try:
                    inside = self.inside[:i] + (first.merge(second), )\
                        + self.inside[i + 2:]
                    self = self.factory(inside, self.dom, self.cod)
                    keep_on_going = True
                    break
                except AxiomError:
                    continue
            if not keep_on_going:
                break
        return self

    def depth(self):
        """
        Computes (an upper bound to) the depth of a diagram by foliating it.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', y, x)
        >>> assert Id(x @ y).depth() == 0
        >>> assert f.depth() == 1
        >>> assert (f @ g).depth() == 1
        >>> assert (f >> g).depth() == 2

        Note
        ----
        The depth of a diagram is the minimum length over all its foliations,
        this method just returns the length of :meth:`Diagram.foliation`.
        """
        return len(self.foliation())

    def interchange(self, i: int, j: int, left=False) -> Diagram:
        """
        Interchange a box from layer ``i`` to layer ``j``.

        Parameters:
            i : Index of the box to interchange.
            j : Index of the new position for the box.
            left : Whether to apply left interchangers.

        Note
        ----
        By default, we apply right interchangers::

            top >> left @ box1.dom @ mid @ box0     @ right\\
                >> left @ box1     @ mid @ box0.cod @ right >> bottom

        gets rewritten to::

            top >> left @ box1     @ mid @ box0.dom @ right\\
                >> left @ box1.cod @ mid @ box0     @ right >> bottom
        """
        if any(len(list(layer)) != 3 for layer in self.inside):
            raise NotImplementedError
        if not 0 <= i < len(self) or not 0 <= j < len(self):
            raise IndexError
        if i == j:
            return self
        if j < i - 1:
            result = self
            for k in range(i - j):
                result = result.interchange(i - k, i - k - 1, left=left)
            return result
        if j > i + 1:
            result = self
            for k in range(j - i):
                result = result.interchange(i + k, i + k + 1, left=left)
            return result
        if j < i:
            i, j = j, i
        off0, off1 = self.offsets[i], self.offsets[j]
        left0, box0, right0 = self.inside[i]
        left1, box1, right1 = self.inside[j]
        # By default, we check if box0 is to the right first, then to the left.
        if left and off1 >= off0 + len(box0.cod):  # box0 left of box1
            off1 = off1 - len(box0.cod) + len(box0.dom)
            middle = left1[len(left0 @ box0.cod):]
            layer0 = left0 @ box0 @ middle @ box1.cod @ right1
            layer1 = left0 @ box0.dom @ middle @ box1 @ right1
        elif off0 >= off1 + len(box1.dom):  # box0 right of box1
            off0 = off0 - len(box1.dom) + len(box1.cod)
            middle = left0[len(left1 @ box1.dom):]
            layer0 = left1 @ box1.cod @ middle @ box0 @ right0
            layer1 = left1 @ box1 @ middle @ box0.dom @ right0
        elif off1 >= off0 + len(box0.cod):  # box0 left of box1
            off1 = off1 - len(box0.cod) + len(box0.dom)
            middle = left1[len(left0 @ box0.cod):]
            layer0 = left0 @ box0 @ middle @ box1.cod @ right1
            layer1 = left0 @ box0.dom @ middle @ box1 @ right1
        else:
            raise AxiomError(messages.INTERCHANGER_ERROR.format(box0, box1))
        return self[:i] >> layer1 >> layer0 >> self[i + 2:]

    def normalize(self, left=False) -> Iterator[Diagram]:
        """
        Implements normalisation of boundary-connected diagrams,
        see Delpeuch and Vicary :cite:t:`DelpeuchVicary22`.

        Parameters:
            left : Passed to :meth:`Diagram.interchange`.

        Example
        -------
        >>> from discopy.monoidal import *
        >>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
        >>> gen = (s0 @ s1).normalize()
        >>> for _ in range(3): print(next(gen))
        s1 >> s0
        s0 >> s1
        s1 >> s0
        """
        diagram = self
        while True:
            no_more_moves = True
            for i in range(len(diagram) - 1):
                box0, box1 = diagram.boxes[i], diagram.boxes[i + 1]
                off0, off1 = diagram.offsets[i], diagram.offsets[i + 1]
                if left and off1 >= off0 + len(box0.cod)\
                        or not left and off0 >= off1 + len(box1.dom):
                    diagram = diagram.interchange(i, i + 1, left=left)
                    yield diagram
                    no_more_moves = False
            if no_more_moves:
                break

    def normal_form(self, **params) -> Diagram:
        """
        Returns the normal form of a diagram.

        params : Passed to :meth:`Diagram.normalize`.

        Raises
        ------
        NotImplementedError
            Whenever ``normalize`` yields the same rewrite steps twice, e.g.
            the diagram is not boundary-connected.
        """
        cache = set()
        for diagram in itertools.chain([self], self.normalize(**params)):
            if str(diagram) in cache:
                exception = NotImplementedError(
                    messages.NOT_CONNECTED.format(self))
                exception.last_step = diagram
                raise exception
            cache.add(str(diagram))
        return diagram

    @classmethod
    def from_tree(cls, tree):
        if "inside" not in tree:
            warn("Outdated dumps", DeprecationWarning)
            boxes, offsets = map(from_tree, tree['boxes']), tree['offsets']
            return cls.decode(from_tree(tree['dom']), zip(boxes, offsets))
        return super().from_tree(tree)


class Box(cat.Box, Diagram):
    """
    A box is a diagram with a :code:`name` and the layer of just itself inside.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. the input.
        cod : The codomain of the box, i.e. the output.
        data (any) : Extra data in the box, default is :code:`None`.
        is_dagger (bool, optional) : Whether the box is dagger.

    Other parameters
    ----------------

    draw_as_spider : bool, optional
        Whether to draw the box as a spider.
    draw_as_wires : bool, optional
        Whether to draw the box as wires, e.g. :class:`discopy.symmetric.Swap`.
    draw_as_braid : bool, optional
        Whether to draw the box as a a braid, e.g. :class:`braided.Braid`.
    drawing_name : str, optional
        The name to use when drawing the box.
    tikzstyle_name : str, optional
        The name of the style when tikzing the box.
    color : str, optional
        The color to use when drawing the box, one of
        :code:`"white", "red", "green", "blue", "yellow", "black"`.
        Default is :code:`"red" if draw_as_spider else "white"`.
    shape : str, optional
        The shape to use when drawing a spider,
        one of :code:`"circle", "rectangle"`.

    Examples
    --------
    >>> f = Box('f', Ty('x', 'y'), Ty('z'))
    >>> assert Id(Ty('x', 'y')) >> f == f == f >> Id(Ty('z'))
    >>> assert Id(Ty()) @ f == f == f @ Id(Ty())
    >>> assert f == f[::-1][::-1]
    """
    __ambiguous_inheritance__ = (cat.Box, )

    def to_drawing(self) -> Box:
        dom, cod = self.dom.to_drawing(), self.cod.to_drawing()
        result = Box(self.name, dom, cod, is_dagger=self.is_dagger)
        for attr, default in drawing.ATTRIBUTES.items():
            setattr(result, attr, getattr(self, attr, default(result)))
        return result

    def __init__(self, name: str, dom: Ty, cod: Ty, **params):
        for attr in drawing.ATTRIBUTES:
            value = params.pop(attr, None)
            if value is not None:
                setattr(self, attr, value)
        cat.Box.__init__(self, name, dom, cod, **params)
        inside = (self.layer_factory.cast(self), )
        Diagram.__init__(self, inside, dom, cod)


class Sum(cat.Sum, Box):
    """
    A sum is a tuple of diagrams :code:`terms`
    with the same domain and codomain.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.

    Example
    -------
    >>> f = Box('f', 'x', 'x')
    >>> print(f @ (f + f))
    (f @ x >> x @ f) + (f @ x >> x @ f)
    """
    __ambiguous_inheritance__ = (cat.Sum, )

    def tensor(self, other=None, *others):
        if other is None or others:
            return Diagram.tensor(self, other, *others)
        other = other if isinstance(other, Sum)\
            else self.sum_factory((other, ))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        terms = tuple(f.tensor(g) for f in self.terms for g in other.terms)
        return self.sum_factory(terms, dom, cod)

    def draw(self, **params):
        """ Drawing a sum as an equation with :code:`symbol='+'`. """
        return drawing.Equation(*self.terms, symbol='+').draw(**params)


class Bubble(cat.Bubble, Box):
    """
    A bubble is a box with a diagram :code:`arg` inside and an optional pair of
    types :code:`dom` and :code:`cod`.

    Parameters:
        arg : The diagram inside the bubble.
        dom : The domain of the bubble, default is that of :code:`other`.
        cod : The codomain of the bubble, default is that of :code:`other`.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x, y ** 3), Box('g', y, y @ y)
    >>> d = (f.bubble(dom=x @ x, cod=y) >> g).bubble()
    >>> d.draw(path='docs/_static/monoidal/bubble-example.png')

    .. image:: /_static/monoidal/bubble-example.png
        :align: center
    """
    __ambiguous_inheritance__ = (cat.Bubble, )

    def __init__(self, arg: Diagram, dom: Ty = None, cod: Ty = None, **params):
        self.drawing_name = params.get("drawing_name", "")
        cat.Bubble.__init__(self, arg, dom, cod)
        Box.__init__(self, self.name, self.dom, self.cod, data=self.data)

    def to_drawing(self):
        dom, cod = self.dom.to_drawing(), self.cod.to_drawing()
        argdom, argcod = self.arg.dom.to_drawing(), self.arg.cod.to_drawing()
        left, right = Ty(self.drawing_name), Ty("")
        left.inside[0].always_draw_label = True
        _open = Box("_open", dom, left @ argdom @ right).to_drawing()
        _close = Box("_close", left @ argcod @ right, cod).to_drawing()
        _open.draw_as_wires = _close.draw_as_wires = True
        # Wires can go straight only if types have the same length.
        _open.bubble_opening = len(dom) == len(argdom)
        _close.bubble_closing = len(cod) == len(argcod)
        return _open >> left @ self.arg.to_drawing() @ right >> _close


class Category(cat.Category):
    """
    A monoidal category is a category with a method :code:`tensor`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    __ambiguous_inheritance__ = True

    ob, ar = Ty, Diagram


class Functor(cat.Functor):
    """
    A monoidal functor is a functor that preserves the tensor product.

    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.

    Important
    ---------
    The keys of the objects mapping must be atomic types, i.e. of length 1.

    Example
    -------
    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y, data=[0.1]), Box('f1', z, w, data=[1.1])
    >>> F = Functor({x: z, y: w, z: x, w: y}, {f0: f1, f1: f0})
    >>> assert F(f0) == f1 and F(f1) == f0
    >>> assert F(F(f0)) == f0
    >>> assert F(f0 @ f1) == f1 @ f0
    >>> assert F(f0 >> f0[::-1]) == f1 >> f1[::-1]
    >>> source, target = f0 >> f0[::-1], F(f0 >> f0[::-1])

    >>> from discopy.drawing import Equation
    >>> Equation(source, target, symbol='$\\\\mapsto$').draw(
    ...     figsize=(4, 2), path='docs/_static/monoidal/functor-example.png')

    .. image:: /_static/monoidal/functor-example.png
        :align: center
    """
    __ambiguous_inheritance__ = True

    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, PRO):
            return sum(other.n * [self.ob[other.factory(1)]], self.cod.ob())
        if isinstance(other, Ty):
            return sum(map(self, other.inside), self.cod.ob())
        if isinstance(other, cat.Ob):
            result = self.ob[self.dom.ob(other)]
            cod_type = getattr(self.cod.ob, "__origin__", self.cod.ob)
            # Syntactic sugar {x: n} in tensor and {x: int} in python.
            return result if isinstance(result, cod_type) else\
                (result, ) if cod_type == tuple else self.cod.ob(result)
        if isinstance(other, Layer):
            head, *tail = other
            result = self(head)
            for box_or_typ in tail:
                result = result @ self(box_or_typ)
            return result
        return super().__call__(other)


@dataclass
class Match:
    """ A match is a diagram with a hole, given by:

    Parameters:
        above : The diagram above the hole.
        below : The diagram below the hole.
        left : The wires left of the hole.
        right : The wires right of the hole.
    """

    above: Diagram
    below: Diagram
    left: Ty
    right: Ty

    def subs(self, target: Diagram) -> Diagram:
        """
        Substitute a diagram inside the hole.

        Parameters:
            target : The diagram to substitute inside the hole.
        """
        return self.above >> self.left @ target @ self.right >> self.below


class Hypergraph(hypergraph.Hypergraph):
    category, functor = Category, Functor

    def to_diagram(self):
        if not self.is_monogamous:
            raise AxiomError(factory_name(
                self.category.ar) + " does not have copy or discard.")
        return super().to_diagram()


Diagram.draw = drawing.draw
Diagram.to_gif = drawing.to_gif
Diagram.to_grid = drawing.Grid.from_diagram

Diagram.sum_factory = Sum
Diagram.bubble_factory = Bubble
Diagram.hypergraph_factory = Hypergraph
Id = Diagram.id
