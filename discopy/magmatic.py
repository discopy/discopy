# -*- coding: utf-8 -*-

"""
The free magmatic category, i.e. planar diagrams with non-strict tensor.

Wires can be labeled with explicit tensor products, i.e. :class:`Tensor`
objects which are binary trees with types as leaves. The boxes :class:`Pack`
and :class:`Unpack` merge and split these products into their components.
Diagrams themselves also have an explicit tensor product :class:`Pair`, which
represents two diagrams side by side without decomposing them in terms of
whiskering and composition.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Tensor
    Diagram
    Box
    Pack
    Unpack
    Pair
    Sum
    Functor

Axioms
------

:class:`Pack` and :class:`Unpack` are inverses.

>>> x, y = Ty('x'), Ty('y')
>>> assert Pack(x, y).dagger() == Unpack(x, y)
>>> assert Unpack(x, y).dagger() == Pack(x, y)

:meth:`Diagram.to_monoidal` sends them both to the identity, so that packing
and unpacking is indeed inverse in the image of any strict monoidal category.

>>> from discopy import monoidal
>>> assert (Pack(x, y) >> Unpack(x, y)).to_monoidal()\\
...     == monoidal.Id(monoidal.Ty('x', 'y'))\\
...     == Diagram.id(x @ y).to_monoidal()

The explicit tensor product of diagrams is flattened to the usual one, defined
in terms of whiskering and composition.

>>> f, g = Box('f', x, y), Box('g', y, x)
>>> f_, g_ = [box.to_monoidal() for box in (f, g)]
>>> assert (f & g).to_monoidal() == f_ @ g_
>>> assert Diagram.from_monoidal(f_ @ g_) == f @ g

The explicit tensor product of diagrams is drawn side by side.

>>> (f & g).draw(path='docs/_static/magmatic/pair.png')

.. image:: /_static/magmatic/pair.png
    :align: center
"""

from __future__ import annotations

from discopy import cat, monoidal
from discopy.cat import ob_factory, ar_factory
from discopy.drawing import Drawing
from discopy.utils import assert_isinstance, factory_name, from_tree


@ob_factory
class Ty(monoidal.Ty):
    """
    A magmatic type is a monoidal type whose objects may be explicit
    :class:`Tensor` products, i.e. binary trees with types as leaves.

    Parameters:
        inside (cat.Ob) : The objects inside the type.

    Example
    -------
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> print(x & (y & z))
    (x & (y & z))
    >>> assert len(x & (y & z)) == 1 and (x & (y & z)).flatten() == x @ y @ z
    """
    def pack(self, other: Ty) -> Ty:
        """
        The explicit tensor product with another type, called with ``&``.

        Parameters:
            other : The other type to pack.
        """
        return self.ob(self.tensor_factory(self, other))

    __and__ = pack

    @property
    def is_tensor(self) -> bool:
        """
        Whether the type is an explicit :class:`Tensor` product.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x & y).is_tensor and not (x @ y).is_tensor
        """
        return len(self) == 1 and isinstance(self.inside[0], Tensor)

    @property
    def left(self) -> Ty:
        """ The left-hand side of a tensor, assumes ``self.is_tensor``. """
        assert self.is_tensor
        return self.inside[0].left

    @property
    def right(self) -> Ty:
        """ The right-hand side of a tensor, assumes ``self.is_tensor``. """
        assert self.is_tensor
        return self.inside[0].right

    def unpack(self) -> Ty:
        """
        Split one level of tensor, assumes ``self.is_tensor``.

        Example
        -------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert (x & (y & z)).unpack() == x @ (y & z)
        """
        return self.left @ self.right

    def flatten(self) -> Ty:
        """
        Split all the explicit tensor products recursively.

        Example
        -------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert ((x & y) & z).flatten() == x @ y @ z
        >>> assert (x @ y @ z).flatten() == x @ y @ z
        """
        return self.ob().tensor(*[
            x.flatten() if isinstance(x, Tensor) else self.ob(x)
            for x in self.inside])


class Tensor(cat.Ob):
    """
    The explicit tensor product of a pair of types as a single object,
    i.e. a binary tree with objects as leaves.

    Parameters:
        left : The left-hand side of the tensor.
        right : The right-hand side of the tensor.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> tensor = Tensor(x, y)
    >>> assert x & y == Ty(tensor) and (x & y).left == x
    """

    ob = Ty

    def __init__(self, left: Ty, right: Ty):
        assert_isinstance(left, self.ob)
        assert_isinstance(right, self.ob)
        self.left, self.right = left, right
        super().__init__(str(self))

    def __eq__(self, other):
        return isinstance(other, type(self))\
            and (self.left, self.right) == (other.left, other.right)

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return f"({self.left} & {self.right})"

    def __repr__(self):
        return factory_name(type(self)) + f"({self.left!r}, {self.right!r})"

    def flatten(self) -> Ty:
        """ The tensor of the flattened left- and right-hand sides. """
        return self.left.flatten() @ self.right.flatten()

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'left': self.left.to_tree(), 'right': self.right.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['left'], tree['right'])))


@ar_factory
class Diagram(monoidal.Diagram):
    """
    A magmatic diagram is a monoidal diagram with :class:`Pack` and
    :class:`Unpack` boxes and an explicit tensor product :class:`Pair`.

    Parameters:
        inside (monoidal.Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty

    @classmethod
    def pack(cls, left: Ty, right: Ty) -> Diagram:
        """
        Wrapper around :class:`Pack` called by :class:`Functor`.

        Parameters:
            left : The left-hand side of the tensor.
            right : The right-hand side of the tensor.
        """
        return cls.pack_factory(left, right)

    @classmethod
    def unpack(cls, left: Ty, right: Ty) -> Diagram:
        """
        Wrapper around :class:`Unpack` called by :class:`Functor`.

        Parameters:
            left : The left-hand side of the tensor.
            right : The right-hand side of the tensor.
        """
        return cls.unpack_factory(left, right)

    @classmethod
    def pair(cls, left: Diagram, right: Diagram,
             dom: Ty = None, cod: Ty = None) -> Diagram:
        """
        Wrapper around :class:`Pair` called by :class:`Functor`.

        Parameters:
            left : The diagram on the left of the pair.
            right : The diagram on the right of the pair.
            dom : The domain of the pair, ``left.dom & right.dom`` by default.
            cod : The codomain, ``left.cod & right.cod`` by default.
        """
        return cls.pair_factory(left, right, dom=dom, cod=cod)

    def __and__(self, other):
        other = other if isinstance(other, Diagram) else self.id(other)
        return self.pair(self, other)

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)

    def to_monoidal(self) -> monoidal.Diagram:
        """
        Flatten a magmatic diagram into a strict monoidal one, i.e. split all
        the explicit tensor products into their components, send :class:`Pack`
        and :class:`Unpack` to identities and :class:`Pair` to
        :meth:`monoidal.Diagram.tensor`.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', Ty(), x & y)
        >>> print((f >> Unpack(x, y)).to_monoidal())
        f
        """
        def ob(typ: Ty) -> monoidal.Ty:
            return monoidal.Ty(*typ.inside)

        ob_functor = Functor(ob, {}, cod=monoidal.Diagram)

        def ar(box: Box) -> monoidal.Box:
            return monoidal.Box(
                box.name, ob_functor(box.dom), ob_functor(box.cod))

        return Functor(ob, ar, cod=monoidal.Diagram)(self)

    @classmethod
    def from_monoidal(cls, diagram: monoidal.Diagram) -> Diagram:
        """
        Turn a strict monoidal diagram into a magmatic one.

        Parameters:
            diagram : The monoidal diagram to embed.

        Example
        -------
        >>> from discopy import monoidal
        >>> f = monoidal.Box('f', monoidal.Ty('x'), monoidal.Ty('y'))
        >>> assert Diagram.from_monoidal(f) == Box('f', Ty('x'), Ty('y'))
        """
        def ob(typ: monoidal.Ty) -> Ty:
            return cls.ob(*typ.inside)

        ob_functor = monoidal.Functor(ob, {}, cod=cls)

        def ar(box: monoidal.Box) -> Box:
            return Box(box.name, ob_functor(box.dom), ob_functor(box.cod))

        return monoidal.Functor(ob, ar, cod=cls)(diagram)


class Box(monoidal.Box, Diagram):
    """
    A magmatic box is a monoidal box in a magmatic diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Pack(Box):
    """
    The box that merges two wires of type ``left`` and ``right`` into a single
    wire of type ``left & right``.

    Parameters:
        left : The left-hand side of the tensor.
        right : The right-hand side of the tensor.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Pack(x, y).dom == x @ y and Pack(x, y).cod == x & y
    """
    def __init__(self, left: Ty, right: Ty):
        assert_isinstance(left, Ty)
        assert_isinstance(right, Ty)
        self.left, self.right = left, right
        name = f"Pack({left}, {right})"
        super().__init__(name, left @ right, left & right,
                         draw_as_spider=True, color="black", drawing_name="")

    def dagger(self) -> Unpack:
        return self.unpack_factory(self.left, self.right)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.left!r}, {self.right!r})"

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'left': self.left.to_tree(), 'right': self.right.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['left'], tree['right'])))


class Unpack(Box):
    """
    The box that splits a single wire of type ``left & right`` into two wires
    of type ``left`` and ``right``, i.e. the dagger of :class:`Pack`.

    Parameters:
        left : The left-hand side of the tensor.
        right : The right-hand side of the tensor.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> assert Unpack(x, y).dom == x & y and Unpack(x, y).cod == x @ y
    """
    def __init__(self, left: Ty, right: Ty):
        assert_isinstance(left, Ty)
        assert_isinstance(right, Ty)
        self.left, self.right = left, right
        name = f"Unpack({left}, {right})"
        super().__init__(name, left & right, left @ right,
                         draw_as_spider=True, color="black", drawing_name="")

    def dagger(self) -> Pack:
        return self.pack_factory(self.left, self.right)

    __repr__, to_tree = Pack.__repr__, Pack.to_tree

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['left'], tree['right'])))


class Pair(monoidal.Bubble, Box):
    """
    The explicit tensor product of two diagrams ``left & right``, i.e. a single
    box with the packed domains and codomains, called with ``&``.

    Parameters:
        left : The diagram on the left of the pair.
        right : The diagram on the right of the pair.
        dom : The domain of the pair, ``left.dom & right.dom`` by default.
        cod : The codomain of the pair, ``left.cod & right.cod`` by default.

    Example
    -------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x, y), Box('g', y, x)
    >>> assert f & g == Pair(f, g)
    >>> assert (f & g).dom == x & y and (f & g).cod == y & x
    """
    def __init__(self, left: Diagram, right: Diagram,
                 dom: Ty = None, cod: Ty = None):
        dom = left.dom & right.dom if dom is None else dom
        cod = left.cod & right.cod if cod is None else cod
        monoidal.Bubble.__init__(
            self, left, right, dom=dom, cod=cod, method="pair")

    @property
    def left(self) -> Diagram:
        """ The diagram on the left of the pair. """
        return self.args[0]

    @property
    def right(self) -> Diagram:
        """ The diagram on the right of the pair. """
        return self.args[1]

    def decompose(self) -> Diagram:
        """
        Decompose the pair as unpacking, tensor and packing.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', y, x)
        >>> assert (f & g).decompose()\\
        ...     == Unpack(x, y) >> f @ g >> Pack(y, x)
        """
        return self.unpack_factory(self.left.dom, self.right.dom)\
            >> self.left @ self.right\
            >> self.pack_factory(self.left.cod, self.right.cod)

    def dagger(self) -> Pair:
        return type(self)(self.left.dagger(), self.right.dagger(),
                          dom=self.cod, cod=self.dom)

    def __str__(self):
        return f"({self.left} & {self.right})"


class Sum(monoidal.Sum, Box):
    """
    A magmatic sum is a monoidal sum of magmatic diagrams.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(monoidal.Functor):
    """
    A magmatic functor is a monoidal functor that preserves explicit tensor
    products, packing and unpacking.

    Parameters:
        ob (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.

    Note
    ----
    If the codomain is strict, i.e. it has no method ``pack``, then the
    explicit tensor products are flattened: :class:`Tensor` objects are sent
    to the tensor of their components, :class:`Pack` and :class:`Unpack` to
    identities and :class:`Pair` to the tensor product of diagrams.

    Example
    -------
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> f = Box('f', x, y & z)
    >>> F = Functor({x: y, y: z, z: x}, {f: Box('f', y, z & x)})
    >>> assert F(x & y) == y & z and F(f.dom) == y
    >>> assert F(f >> Unpack(y, z)) == Box('f', y, z & x) >> Unpack(z, x)
    """
    dom = cod = Diagram

    def __call__(self, other):
        if self.cod is Drawing:
            return super().__call__(other)
        if isinstance(other, Tensor):
            left, right = self(other.left), self(other.right)
            return left.pack(right) if hasattr(left, "pack")\
                else left @ right
        if isinstance(other, (Pack, Unpack)):
            left, right = self(other.left), self(other.right)
            if hasattr(self.cod, "pack"):
                result = self.cod.pack(left, right)
                return result.dagger() if isinstance(other, Unpack)\
                    else result
            return self.cod.id(left @ right)
        if isinstance(other, Pair) and not hasattr(self.cod, "pair"):
            return self(other.left) @ self(other.right)
        return super().__call__(other)


Diagram.pack_factory = Pack
Diagram.unpack_factory = Unpack
Diagram.pair_factory = Pair
Diagram.sum_factory = Sum
Ty.tensor_factory = Tensor

Id = Diagram.id
