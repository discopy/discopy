# -*- coding: utf-8 -*-

"""
The free (pre)monoidal category, i.e. planar diagrams.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Colour
    Wire
    Ty
    PRO
    Dim
    Layer
    Diagram
    Box
    Sum
    Bubble
    Functor

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
from dataclasses import dataclass
from typing import Iterator, Callable, TYPE_CHECKING
from warnings import warn

from discopy import cat, drawing, hypergraph, cmap, messages
from discopy.abc import ColouredMonoid, MonoidalCategory
from discopy.drawing import Drawing
from discopy.config import BOX_DRAWING_ATTRIBUTES, WIRE_DRAWING_ATTRIBUTES
from discopy.utils import (
    factory,
    factory_name,
    from_tree,
    assert_isinstance,
    assert_iscomposable,
    AxiomError,
    get_origin,
    MappingOrCallable,
)

if TYPE_CHECKING:
    import sympy


@dataclass(frozen=True)
class Colour(cat.Ob):
    """A 0-cell, drawn using its matplotlib-compatible name."""

    name: str = "white"

    def __post_init__(self):
        assert_isinstance(self.name, str)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.name!r})"


white = Colour("white")


class Wire(cat.Ob):
    """A generating 1-cell with a colour on either side."""

    def __init__(self, name: str, dom: Colour = white,
                 cod: Colour = white, is_dagger: bool = False):
        assert_isinstance(dom, Colour)
        assert_isinstance(cod, Colour)
        self.is_dagger = is_dagger
        self.dom, self.cod = dom, cod
        super().__init__(name)

    def __setstate__(self, state):
        state.setdefault('dom', white)
        state.setdefault('cod', white)
        state.setdefault('is_dagger', False)
        super().__setstate__(state)

    def dagger(self):
        return type(self)(
            self.name, self.cod, self.dom, is_dagger=not self.is_dagger)

    def __eq__(self, other):
        return type(self) is type(other) and (
            self.name, self.dom, self.cod) == (
                other.name, other.dom, other.cod)

    def __hash__(self):
        return hash((type(self), self.name, self.dom, self.cod))

    def __repr__(self):
        if self.dom == self.cod == white:
            return repr(cat.Ob(self.name))
        return (f"{factory_name(type(self))}({self.name!r}, "
                f"dom={self.dom!r}, cod={self.cod!r})")

    def to_tree(self):
        tree = super().to_tree()
        tree['factory'] = factory_name(type(self))
        if self.dom != white:
            tree['dom'] = self.dom.to_tree()
        if self.cod != white:
            tree['cod'] = self.cod.to_tree()
        if self.is_dagger:
            tree['is_dagger'] = True
        return tree

    @classmethod
    def from_tree(cls, tree):
        dom = from_tree(tree['dom']) if 'dom' in tree else white
        cod = from_tree(tree['cod']) if 'cod' in tree else white
        return cls(tree['name'], dom, cod, is_dagger='is_dagger' in tree)


class FreeMonoid(cat.FreeCategory, ColouredMonoid):
    """A free category whose composition is also its monoid product."""

    def __init__(self, inside, dom: Colour = None, cod: Colour = None,
                 _scan: bool = True):
        if dom is None:
            dom = inside[0].dom if inside else white
        if cod is None:
            cod = inside[-1].cod if inside else white
        cat.FreeCategory.__init__(self, inside, dom, cod, _scan)

    def tensor(self, *others):
        # Whiskering: tensoring a type with e.g. a diagram returns
        # NotImplemented so the other operand's __rmatmul__ takes over.
        if any(not isinstance(other, self.factory) for other in others):
            return NotImplemented
        return cat.FreeCategory.then(self, *others)

    then = tensor


@factory
class Ty(cat.Ob, FreeMonoid):
    """
    A type is a composable path of objects with :meth:`Ty.tensor`
    as concatenation.

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

    Tip
    ---
    Types can also be instantiated by keyword, passing the path of
    generators as ``inside=``; this is what the free-category machinery
    uses internally, while the variadic form above is the user-friendly
    ``Ty('x', 'y')`` API.

    >>> assert Ty(inside=(Wire('x'), Wire('y'))) == Ty('x', 'y')

    Note
    ----
    Types can be indexed and sliced using square brackets. Indexing behaves
    like that of strings, i.e. when we index a type we get a type back.
    The objects inside the type are still accessible using ``.inside``.

    >>> t = Ty(*"xyz")
    >>> assert t[0] == t[:1] == Ty('x')
    >>> assert t[0] != t.inside[0] == Wire('x')
    >>> assert t[1:] == t[-2:] == Ty('y', 'z')
    """
    ob = Colour
    generator_factory = Wire

    def cast_wire(self, x: str | cat.Ob) -> cat.Ob:
        """
        Turn a constructor argument into a ``self.generator_factory``.

        Old dumps and pickles used a plain ``cat.Ob``, with no colour, as
        the generators: upgrade it to ``Wire(x.name)`` for subclasses whose
        generators are plain ``Wire``.
        """
        if isinstance(x, self.generator_factory):
            return x
        if isinstance(x, str):
            return self.generator_factory(x)
        if self.generator_factory is Wire and type(x) is cat.Ob:
            return self.generator_factory(x.name)
        raise AxiomError(
            messages.TYPE_ERROR.format(self.generator_factory, type(x)))

    def __init__(self, *inside: str | cat.Ob,
                 dom: Colour = None, cod: Colour = None,
                 _scan: bool = True, **kwargs):
        inside = kwargs.pop('inside', inside)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}.")
        for obj in inside:
            assert_isinstance(obj, (str, self.generator_factory) + (
                (cat.Ob, ) if self.generator_factory is Wire else ()))
        inside = tuple(map(self.cast_wire, inside))
        FreeMonoid.__init__(self, inside, dom, cod, _scan)
        cat.Ob.__init__(self, str(self))

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

    @property
    def is_atomic(self) -> bool:
        """ Whether a type is atomic, i.e. it has length 1. """
        return len(self) == 1

    def __eq__(self, other):
        return type(self) is type(other) and self.inside == other.inside\
            and (self.dom, self.cod) == (other.dom, other.cod)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        if not self.inside and self.dom != white:
            return f"{factory_name(type(self))}.id({self.dom!r})"
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.inside))})"

    def __str__(self):
        name = type(self).__name__
        if not self.inside:
            if self.dom == white:
                return f"{name}()"
            return f"{name}.id({self.dom})"
        parts = []
        for ob in self.inside:
            s = str(ob)
            parts.append(f'{name}("")' if s == '' else s)
        return ' @ '.join(parts)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __pow__(self, n_times):
        assert_isinstance(n_times, int)
        if n_times <= 0:
            assert self.dom == self.cod
            return self.factory.id(self.dom)
        return self.tensor(*(n_times - 1) * [self])

    def __setstate__(self, state):
        if 'inside' not in state and "_objects" in state:
            state["inside"] = state['_objects']
            del state['_objects']
        if 'dom' not in state:
            state['dom'] = white
        if 'cod' not in state:
            state['cod'] = white
        self.__dict__.update(state)
        if not hasattr(self, 'name'):
            self.name = str(self)

    def to_tree(self):
        tree = {
            'factory': factory_name(type(self)),
            'inside': [x.to_tree() for x in self.inside]}
        if not self.inside and self.dom != white:
            tree['dom'] = self.dom.to_tree()
            tree['cod'] = self.cod.to_tree()
        return tree

    @classmethod
    def from_tree(cls, tree):
        if "inside" not in tree:
            warn("Outdated dumps", DeprecationWarning)
            return cls(*map(from_tree, tree['objects']))
        inside = tuple(map(from_tree, tree['inside']))
        # Old dumps used cat.Ob as the generators of monoidal.Ty.
        inside = tuple(
            cls.generator_factory(x.name) if type(x) is cat.Ob else x
            for x in inside)
        if inside:
            return cls(*inside)
        if 'dom' in tree:
            return cls(dom=from_tree(tree['dom']), cod=from_tree(tree['cod']))
        return cls()

    __add__ = FreeMonoid.__matmul__

    def to_drawing(self) -> Ty:
        if not self.inside:
            return Ty.id(self.dom)
        result = Ty(*(Wire(str(x), getattr(x, 'dom', white),
                           getattr(x, 'cod', white)) for x in self.inside))
        for new, old in zip(result.inside, self.inside):
            if getattr(old, "frame_boundary", False):
                new.frame_boundary = True
            for attr, default in WIRE_DRAWING_ATTRIBUTES.items():
                setattr(new, attr, getattr(old, attr, default(new)))
        return result

    def wire_offsets(self) -> list:
        """
        The x-position of each wire of the type relative to the first, i.e. the
        sum of the cell widths ``max(1, right_margin)`` of the objects before
        it: each wire takes up at least a unit, more if its label is longer.

        >>> assert Ty('x', 'y').to_drawing().wire_offsets() == [0, 1]
        """
        offsets, total = [], 0
        for ob in self.inside:
            offsets.append(total)
            total += max(1, ob.right_margin)
        return offsets


@factory
class PRO(Ty):
    """
    A PRO is a natural number ``n`` seen as a type with addition as tensor.

    Parameters
    ----------
    inside : int | tuple
        The length of the PRO type, or a tuple of generators whose
        length is taken.

    Example
    -------
    >>> assert PRO(1) @ PRO(2) == PRO(3)

    Note
    ----
    If ``ob`` is ``PRO`` then :class:`Diagram` will automatically turn
    any ``n: int`` into ``PRO(n)``. Thus ``PRO`` never needs to be called.

    >>> @factory
    ... class Circuit(Diagram):
    ...     ob = PRO
    >>> class Gate(Box, Circuit): ...
    >>> CX = Gate('CX', 2, 2)

    >>> assert CX @ 2 >> 2 @ CX == CX @ CX
    """
    def __init__(self, inside: int | tuple = 0, dom: Colour = None,
                 cod: Colour = None, _scan: bool = True):
        self.n = inside if isinstance(inside, int) else len(inside)
        self.dom = self.cod = white
        self.name = str(self)

    def __setstate__(self, state):
        if "n" not in state:
            state = {"n": len(state["_objects"])}
        state.setdefault("dom", white)
        state.setdefault("cod", white)
        state.setdefault("name", f"PRO({state['n']})")
        self.__dict__.update(state)

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

    then = tensor

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

    def to_tree(self):
        return {'factory': factory_name(type(self)), 'n': self.n}

    @classmethod
    def from_tree(cls, tree):
        return cls(tree['n'])


@factory
class Dim(Ty):
    """
    A dimension is a tuple of positive integers
    with product ``@`` and unit ``Dim(1)``.

    Example
    -------
    >>> Dim(1) @ Dim(2) @ Dim(3)
    Dim(2, 3)
    """
    generator_factory = int

    def __init__(self, *inside: int, dom=None, cod=None, _scan=True, **kwargs):
        inside = kwargs.pop('inside', inside)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}.")
        for dim in inside:
            assert_isinstance(dim, int)
            if dim < 1:
                raise ValueError
        inside = tuple(dim for dim in inside if dim > 1)
        cat.FreeCategory.__init__(
            self, inside, white if dom is None else dom,
            white if cod is None else cod, _scan=False)
        cat.Ob.__init__(self, str(self))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.factory(*self.inside[key])
        if key >= len(self) or key < -len(self):
            raise IndexError
        return self.factory(self.inside[key])

    def __repr__(self):
        return f"Dim({', '.join(map(repr, self.inside)) or '1'})"

    __str__ = __repr__


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
    ob = Ty

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

    @property
    def boxes(self):
        return list(self.boxes_or_types[1::2])

    @property
    def size(self):
        return sum(box.size for box in self.boxes)

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

    @property
    def is_generator(self):
        if len(self.boxes_or_types) != 3:
            return False
        left, box, right = self.boxes_or_types
        return not left.inside and not right.inside

    @property
    def generator(self):
        return self.boxes_or_types[1] if self.is_generator else None

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
            diagram = Diagram.normal_form(self.boxes_or_types[1].ar(
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
class Diagram(cat.Arrow, MonoidalCategory):
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
    ob = Ty
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

    @property
    def size(self):
        return sum(box.size for box in self.inside)

    @property
    def is_generator(self):
        """ Whether a `Diagram` is a generator, i.e. a single box. """
        return len(self) == 1 and self.inside[0].is_generator

    @property
    def generator(self):
        """ The single box in a generator `Diagram`. """
        return self.inside[0].generator if self.is_generator else None

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
        >>> snake.draw(path='docs/_static/monoidal/diagramize.png')

        .. image:: /_static/monoidal/diagramize.png
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
        assert_isinstance(other, self.ar)
        assert_isinstance(self, other.ar)
        inside = tuple(layer @ other.dom for layer in self.inside)\
            + tuple(self.cod @ layer for layer in other.inside)
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        return self.ar(inside, dom, cod, _scan=False)

    @property
    def boxes(self) -> list[Box]:
        """ The boxes in each layer of the diagram. """
        return sum([layer.boxes for layer in self.inside], [])

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
        >>> diagram.draw(path='docs/_static/monoidal/arrow-example.png')

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

    def to_drawing(self, functor_factory=None) -> Drawing:
        """ Called before :meth:`Diagram.draw`. """
        ob = ar = lambda x: x.to_drawing()
        dom = self.ar
        cod = Drawing
        return (functor_factory or Functor)(ob, ar, dom, cod)(self)

    def to_map(self) -> CMap:
        """ Translate a diagram into a combinatorial map. """
        return self.map_factory.from_diagram(self)

    def to_staircases(self):
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
        return Functor.id(self.ar)(self)

    def foliation(self):
        """
        Merges layers together to reduce the length of a diagram.

        Example
        -------
        >>> from discopy.monoidal import *
        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> diagram = f0 @ f1.dagger() >> f0.dagger() @ f1
        >>> print(diagram)
        f0 @ x >> y @ f1[::-1] >> f0[::-1] @ y >> x @ f1
        >>> diagram.foliation().draw(
        ...     path='docs/_static/monoidal/foliation-example.png')

        .. image:: /_static/monoidal/foliation-example.png
            :align: center

        Note
        ----
        If one defines a foliation as a sequence of unmergeable layers,
        there may exist many distinct foliations for the same diagram.
        This method scans top to bottom and merges layers eagerly.
        """
        while len(self) > 1:
            keep_on_going = False
            for i, (first, second) in enumerate(zip(
                    self.inside, self.inside[1:])):
                try:
                    inside = self.inside[:i] + (first.merge(second), )\
                        + self.inside[i + 2:]
                    self = self.ar(inside, self.dom, self.cod)
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

    def substitute(self, i: int, other: Diagram) -> Diagram:
        """
        Implements operadic composition of nested diagrams,
        replacing box :code:`i` with diagram :code:`other`.
        See Patterson et al :cite:t:`Patterson21`.

        Parameters:
            i : Index of the box to substitute.
            other : The diagram to substitute with.
        """
        left, _, right = self.inside[i]
        outside = Match(self[:i], self[i + 1:], left, right)
        return outside.substitute(other)

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

    Coloured wires carry a region colour on either side, and a box must be
    globular, i.e. its domain and codomain share the same boundary colours.

    >>> red, green, blue = map(Colour, ("red", "green", "blue"))
    >>> x = Ty(Wire("x", red, green))
    >>> y = Ty(Wire("y", green, blue))
    >>> z = Ty(Wire("z", red, blue))
    >>> coloured = Box("coloured", x @ y, z)
    >>> assert coloured.dom.dom == red == coloured.cod.dom
    >>> assert coloured.dom.cod == blue == coloured.cod.cod
    """

    def __init__(self, name: str, dom: Ty, cod: Ty, **params):
        dom = dom if isinstance(dom, self.ob) else self.ob(dom)
        cod = cod if isinstance(cod, self.ob) else self.ob(cod)
        if (dom.dom, dom.cod) != (cod.dom, cod.cod):
            raise AxiomError(messages.NOT_GLOBULAR.format(
                dom.dom, dom.cod, cod.dom, cod.cod))
        for attr in BOX_DRAWING_ATTRIBUTES:
            if attr in params:
                setattr(self, attr, params.pop(attr))
        cat.Box.__init__(self, name, dom, cod, **params)
        inside = (self.layer_factory.cast(self), )
        Diagram.__init__(self, inside, dom, cod)

    @property
    def size(self):
        return 1

    def to_drawing(self):
        return Drawing.from_box(self)


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

    ob = Ty

    @property
    def size(self):
        return 1

    def tensor(self, other=None, *others):
        if other is None or others:
            return Diagram.tensor(self, other, *others)
        other = other if isinstance(other, Sum)\
            else self.sum_factory((other, ))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        terms = tuple(f.tensor(g) for f in self.terms for g in other.terms)
        return self.sum_factory(terms, dom, cod)

    to_drawing = Diagram.to_drawing


class Bubble(cat.Bubble, Box):
    """
    A bubble is a box with diagrams :code:`args` inside and an optional pair of
    types :code:`dom` and :code:`cod`.

    Parameters:
        args (Diagram) : The diagrams inside the bubble.
        drawing_name (str) : The label to use when drawing, empty by default.
        draw_as_square (bool) : Whether to draw the bubble as a square.
        draw_as_frame (bool) : Whether to draw the bubble as a frame.
        draw_vertically (bool) : Whether to draw the frame slots vertically.
        kwargs : Passed to :class:`cat.Bubble`.

    Raises:
        ValueError : When dom is None but all the args have the same dom.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g, h = Box('f', x, y ** 3), Box('g', y, y @ y), Box('h', x, y)
    >>> d = (f.bubble(dom=x ** 3, cod=y, draw_as_square=True) >> g).bubble()
    >>> d.draw(path='docs/_static/monoidal/bubble-example.png')

    .. image:: /_static/monoidal/bubble-example.png
        :align: center

    >>> b = Bubble(f, g, h >> h[::-1], dom=x, cod=y @ y)
    >>> b.draw(path='docs/_static/monoidal/bubble-multiple-args.png')

    .. image:: /_static/monoidal/bubble-multiple-args.png
        :align: center

    >>> b = Bubble(f, g, h, dom=x, cod=y @ y, draw_vertically=True)
    >>> b.draw(path='docs/_static/monoidal/frame-vertical-args.png')

    .. image:: /_static/monoidal/frame-vertical-args.png
        :align: center
    """

    ob = Ty

    def __init__(
            self, *args: Diagram,
            drawing_name: str = None,
            draw_as_frame: bool = None,
            draw_as_square: bool = None,
            draw_vertically=False, **kwargs):
        cat.Bubble.__init__(self, *args, **kwargs)
        Box.__init__(self, self.name, self.dom, self.cod)
        self.drawing_name = "" if drawing_name is None else drawing_name
        self.draw_vertically = draw_vertically
        can_draw_as_square = len(args) == 1
        can_draw_as_bubble = (can_draw_as_square
                              and len(self.dom) == len(self.arg.dom)
                              and len(self.cod) == len(self.arg.cod))
        if len(args) == 1:
            can_draw_as_bubble = (len(self.dom), len(self.cod)) == (
                len(self.arg.dom), len(self.arg.cod))
            self.draw_as_square = draw_as_square or not can_draw_as_bubble
            self.draw_as_frame = draw_as_frame or (
                not can_draw_as_bubble and not self.draw_as_square)
        else:
            self.draw_as_frame = True
            self.draw_as_square = False

    @property
    def size(self):
        """ The number of boxes in a bubble, counting its arguments. """
        return 1 + sum(arg.size for arg in self.args)

    def to_drawing(self):
        method = "frame" if self.draw_as_frame else "bubble"
        args = [arg.to_drawing() for arg in self.args]
        kwargs = dict(
            dom=self.dom.to_drawing(),
            cod=self.cod.to_drawing(),
            name=self.drawing_name)
        if self.draw_as_frame:
            kwargs['draw_vertically'] = self.draw_vertically
        else:
            kwargs['draw_as_square'] = self.draw_as_square
        return getattr(Drawing, method)(*args, **kwargs)


class Functor(cat.Functor):
    """
    A monoidal functor is a functor that preserves the tensor product.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
        colour_map (Mapping[Colour, Colour]) :
            Map from region :class:`Colour` to :code:`cod` colour.

    Important
    ---------
    The keys of the objects mapping must be atomic types, i.e. of length 1.

    Note
    ----
    Colour maps are expected to send colours to colours, so the image of an
    empty coloured identity ``Ty.id(c)`` keeps its (mapped) colour whenever
    ``cod.ob`` has an ``id`` method, e.g. ``F(Ty.id(c)) == Ty.id(F(c))``.

    Example
    -------
    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y, data=0.1), Box('f1', z, w, data=1.1)
    >>> F = Functor({x: z, y: w, z: x, w: y}, {f0: f1, f1: f0})
    >>> assert F(f0) == f1 and F(f1) == f0
    >>> assert F(F(f0)) == f0
    >>> assert F(f0 @ f1) == f1 @ f0
    >>> assert F(f0 >> f0[::-1]) == f1 >> f1[::-1]
    >>> source, target = f0 >> f0[::-1], F(f0 >> f0[::-1])

    >>> from discopy.drawing import Equation
    >>> Equation(source, target, symbol='$\\\\mapsto$').draw(
    ...     path='docs/_static/monoidal/functor-example.png')

    .. image:: /_static/monoidal/functor-example.png
        :align: center
    """

    dom = cod = Diagram

    def __init__(
            self, ob_map=None, ar_map=None,
            dom=None, cod=None, colour_map=None):
        super().__init__(ob_map, ar_map, dom=dom, cod=cod)
        self.colour_map = MappingOrCallable(colour_map or {})

    @classmethod
    def id(cls, dom=None):
        return cls(lambda x: x, lambda f: f, dom=dom, cod=dom)

    def then(self, other):
        assert_isinstance(other, Functor)
        assert_iscomposable(self, other)
        return type(self)(
            self.ob_map.then(other), self.ar_map.then(other),
            colour_map=self.colour_map.then(other) if self.colour_map
            else other.colour_map,
            dom=self.dom, cod=other.cod)

    def __eq__(self, other):
        return super().__eq__(other) and self.colour_map == other.colour_map

    def __repr__(self):
        result = super().__repr__()
        if not self.colour_map:
            return result
        suffix = ')' if result.endswith(')') else ''
        return result[:-len(suffix) if suffix else None] + (
            f", colour_map={self.colour_map!r}{suffix}")

    def _map_colour(self, colour):
        return self.colour_map[colour] if self.colour_map else colour

    def _map_atomic(self, key):
        result = self.ob_map[key]
        cod_type = get_origin(self.cod.ob)
        return result if isinstance(result, cod_type) else\
            (result, ) if cod_type == tuple else self.cod.ob(result)

    def __call__(self, other):
        if isinstance(other, Colour):
            return self._map_colour(other)
        if isinstance(other, PRO):
            result = self._map_atomic(other.factory(1))
            return sum(other.n * [result], self.cod.ob())
        if isinstance(other, Dim):
            return sum([self.ob_map[x] for x in other], self.cod.ob())
        if isinstance(other, Ty):
            if not other.inside:
                # Empty coloured identity: keep its (mapped) boundary colour.
                if not hasattr(self.cod.ob, 'id'):
                    return self.cod.ob()
                return self.cod.ob.id(self(other.dom))
            images = list(map(self, other.inside))
            result = images[0]
            for image in images[1:]:
                result = result + image
            return result
        if isinstance(other, self.dom.ob.generator_factory):
            if isinstance(other, Wire) and other.is_dagger:
                # Map a daggered coloured generator functorially: its image is
                # the dagger of the image of the underlying generator.
                return self(other.dagger()).dagger()
            result = self._map_atomic(self.dom.ob(other))
            if isinstance(other, Wire) and isinstance(result, Ty):
                expected = self(other.dom), self(other.cod)
                if (result.dom, result.cod) != expected:
                    raise AxiomError(messages.NOT_GLOBULAR.format(
                        result.dom, result.cod, *expected))
            return result
        if isinstance(other, Layer):
            head, *tail = other
            result = self(head)
            for box_or_typ in tail:
                result = result @ self(box_or_typ)
            return result
        if isinstance(other, Bubble) and self.cod is Drawing:
            return other.to_drawing()
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

    def substitute(self, target: Diagram) -> Diagram:
        """
        Substitute a diagram inside the hole.

        Parameters:
            target : The diagram to substitute inside the hole.
        """
        return self.above >> self.left @ target @ self.right >> self.below


class Hypergraph(hypergraph.Hypergraph):
    functor = Functor


class CMap(cmap.CMap):
    functor = Functor
    require_planar = True
    require_causal = True
    require_oriented = True
    require_connected = True


Diagram.draw = drawing.draw
Diagram.to_gif = drawing.to_gif

Diagram.sum_factory = Sum
Diagram.bubble_factory = Bubble
Diagram.hypergraph_factory = Hypergraph
Diagram.map_factory = CMap
Drawing.ob = Ty
Id = Diagram.id
