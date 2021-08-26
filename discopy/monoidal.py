# -*- coding: utf-8 -*-

"""
Implements the free dagger monoidal category
and strong dagger monoidal functors.

The syntax for diagrams is given by the following grammar::

    diagram ::= Box(name, dom=type, cod=type)
        | diagram[::-1]
        | diagram @ diagram
        | diagram >> diagram
        | Id(type)

where :code:`[::-1]`, :code:`@` and :code:`>>` denote the dagger, tensor and
composition respectively. The syntax for types is given by::

    type ::= Ty(name) | type @ type | Ty()

Notes
-----
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

.. image:: ../_static/imgs/EckmannHilton.gif
    :align: center
"""

from discopy import cat, messages, drawing, rewriting
from discopy.cat import Ob
from discopy.utils import factory_name, from_tree


class Ty(Ob):
    """
    Implements a type as a list of :class:`discopy.cat.Ob`, used as domain and
    codomain for :class:`monoidal.Diagram`.
    Types are the free monoid on objects with product
    :code:`@` and unit :code:`Ty()`.

    Parameters
    ----------
    objects : list of :class:`discopy.cat.Ob`
        List of objects or object names.

    Important
    ---------
    Elements that are not instance of :class:`discopy.cat.Ob` are implicitly
    taken to be the name of an object, i.e.
    :code:`Ty('x', 'y') == Ty(Ob('x'), Ob('y'))`.

    Notes
    -----
    We can check the axioms for a monoid.

    >>> x, y, z, unit = Ty('x'), Ty('y'), Ty('z'), Ty()
    >>> assert x @ unit == x == unit @ x
    >>> assert (x @ y) @ z == x @ y @ z == x @ (y @ z)
    """
    def __init__(self, *objects):
        self._objects = tuple(
            x if isinstance(x, Ob) else Ob(x) for x in objects)
        super().__init__(self)

    @property
    def objects(self):
        """
        List of objects forming a type.

        Note
        ----

        A type may be sliced into subtypes.

        >>> t = Ty('x', 'y', 'z')
        >>> assert t[0] == Ob('x')
        >>> assert t[:1] == Ty('x')
        >>> assert t[1:] == Ty('y', 'z')

        """
        return list(self._objects)

    def tensor(self, *others):
        """
        Returns the tensor of types, i.e. the concatenation of their lists
        of objects. This is called with the binary operator `@`.

        >>> Ty('x') @ Ty('y', 'z')
        Ty('x', 'y', 'z')

        Parameters
        ----------
        other : monoidal.Ty

        Returns
        -------
        t : monoidal.Ty
            such that :code:`t.objects == self.objects + other.objects`.

        Note
        ----
        We can take the sum of a list of type, specifying the unit `Ty()`.

        >>> types = Ty('x'), Ty('y'), Ty('z')
        >>> Ty().tensor(*types)
        Ty('x', 'y', 'z')

        We can take the exponent of a type by any natural number.

        >>> Ty('x') ** 3
        Ty('x', 'x', 'x')

        """
        for other in others:
            if not isinstance(other, Ty):
                raise TypeError(messages.type_err(Ty, other))
        objects = self.objects + [x for t in others for x in t.objects]
        return self.upgrade(Ty(*objects))

    def count(self, obj):
        """
        Counts the occurrence of a given object.

        Parameters
        ----------
        obj : :class:`Ty` or :class:`Ob`
            either a type of length 1 or an object

        Returns
        -------
        n : int
            such that :code:`n == self.objects.count(ob)`.

        Examples
        --------

        >>> x = Ty('x')
        >>> xs = x ** 5
        >>> assert xs.count(x) == xs.count(x[0]) == xs.objects.count(Ob('x'))
        """
        obj, = obj if isinstance(obj, Ty) else (obj, )
        return self._objects.count(obj)

    @staticmethod
    def upgrade(old):
        """ Allows class inheritance for tensor and __getitem__. """
        return old

    def downgrade(self):
        """ Downgrades to :class:`discopy.monoidal.Ty`. """
        return Ty(*self)

    def __eq__(self, other):
        return isinstance(other, Ty) and self._objects == other._objects

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return "Ty({})".format(', '.join(repr(x.name) for x in self._objects))

    def __str__(self):
        return ' @ '.join(map(str, self._objects)) or 'Ty()'

    def __len__(self):
        return len(self._objects)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.upgrade(Ty(*self._objects[key]))
        return self._objects[key]

    def __matmul__(self, other):
        return self.tensor(other)

    def __pow__(self, n_times):
        if not isinstance(n_times, int):
            raise TypeError(messages.type_err(int, n_times))
        result = type(self)()
        for _ in range(n_times):
            result = result @ self
        return result

    def to_tree(self):
        return {
            'factory': factory_name(self),
            'objects': [x.to_tree() for x in self.objects]}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, tree['objects']))


def types(names):
    """ Transforms strings into instances of :class:`discopy.monoidal.Ty`.

    Examples
    --------
    >>> x, y, z = types("x y z")
    >>> x, y, z
    (Ty('x'), Ty('y'), Ty('z'))
    """
    return list(map(Ty, names.split()))


class PRO(Ty):
    """ Implements the objects of a PRO, i.e. a non-symmetric PROP.
    Wraps a natural number n into a unary type Ty(1, ..., 1) of length n.

    Parameters
    ----------
    n : int
        Number of wires.

    Examples
    --------
    >>> PRO(1) @ PRO(1)
    PRO(2)
    >>> assert PRO(3) == Ty(1, 1, 1)
    >>> assert PRO(1) == PRO(Ob(1))
    """
    @staticmethod
    def upgrade(old):
        for obj in old:
            if obj.name != 1:
                raise TypeError(messages.type_err(int, obj.name))
        return PRO(len(old))

    def __init__(self, n=0):
        if isinstance(n, PRO):
            n = len(n)
        if isinstance(n, Ob):
            n = n.name
        super().__init__(*(n * [1]))

    def __repr__(self):
        return "PRO({})".format(len(self))

    def __str__(self):
        return repr(len(self))


class Layer(cat.Box):
    """
    Layer of a diagram, i.e. a box with wires to the left and right.

    Parameters
    ----------
    left : monoidal.Ty
        Left wires.
    box : monoidal.Box
        Middle box.
    right : monoidal.Ty
        Right wires.

    Examples
    --------
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> f, g = Box('f', y, z), Box('g', z, x)
    >>> Layer(x, f, z)
    Layer(Ty('x'), Box('f', Ty('y'), Ty('z')), Ty('z'))
    >>> first, then = Layer(x, f, z), Layer(x, g, z)
    >>> print(first >> then)
    Id(x) @ f @ Id(z) >> Id(x) @ g @ Id(z)
    """
    def __init__(self, left, box, right):
        self._left, self._box, self._right = left, box, right
        dom, cod = left @ box.dom @ right, left @ box.cod @ right
        super().__init__("Layer", dom, cod)

    def __iter__(self):
        yield self._left
        yield self._box
        yield self._right

    def __repr__(self):
        return "Layer({}, {}, {})".format(
            *map(repr, (self._left, self._box, self._right)))

    def __str__(self):
        left, box, right = self
        return ("{} @ ".format(box.id(left)) if left else "")\
            + str(box)\
            + (" @ {}".format(box.id(right)) if right else "")

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return Layer(self._left, self._box[::-1], self._right)
        return super().__getitem__(key)


class Diagram(cat.Arrow):
    """
    Defines a diagram given dom, cod, a list of boxes and offsets.

    Parameters
    ----------
    dom : monoidal.Ty
        Domain of the diagram.
    cod : monoidal.Ty
        Codomain of the diagram.
    boxes : list of :class:`Diagram`
        Boxes of the diagram.
    offsets : list of int
        Offsets of each box in the diagram.
    layers : list of :class:`Layer`, optional
        Layers of the diagram,
        computed from boxes and offsets if :code:`None`.

    Raises
    ------
    :class:`AxiomError`
        Whenever the boxes do not compose.

    Examples
    --------

    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1, g = Box('f0', x, y), Box('f1', z, w), Box('g', y @ w, y)
    >>> d = Diagram(x @ z, y, [f0, f1, g], [0, 1, 0])
    >>> assert d == f0 @ f1 >> g

    >>> d.draw(figsize=(2, 2),
    ...        path='docs/_static/imgs/monoidal/arrow-example.png')

    .. image:: ../_static/imgs/monoidal/arrow-example.png
        :align: center
    """
    @staticmethod
    def upgrade(old):
        return old

    def downgrade(self):
        """ Downcasting to :class:`discopy.monoidal.Diagram`. """
        dom, cod = Ty(*self.dom), Ty(*self.cod)
        boxes, offsets = [box.downgrade() for box in self.boxes], self.offsets
        return Diagram(dom, cod, boxes, offsets)

    def __init__(self, dom, cod, boxes, offsets, layers=None):
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        if not isinstance(cod, Ty):
            raise TypeError(messages.type_err(Ty, cod))
        if len(boxes) != len(offsets):
            raise ValueError(messages.boxes_and_offsets_must_have_same_len())
        if layers is None:
            layers = cat.Id(dom)
            for box, off in zip(boxes, offsets):
                if not isinstance(box, Diagram):
                    raise TypeError(messages.type_err(Diagram, box))
                if not isinstance(off, int):
                    raise TypeError(messages.type_err(int, off))
                left = layers.cod[:off] if layers else dom[:off]
                right = layers.cod[off + len(box.dom):]\
                    if layers else dom[off + len(box.dom):]
                layers = layers >> self.layer_factory(left, box, right)
            layers = layers >> cat.Id(cod)
        self._layers, self._offsets = layers, tuple(offsets)
        super().__init__(dom, cod, boxes, _scan=False)

    def to_tree(self):
        return dict(cat.Arrow.to_tree(self), offsets=self.offsets)

    @classmethod
    def from_tree(cls, tree):
        arrow = cat.Arrow.from_tree(tree)
        return cls(arrow.dom, arrow.cod, arrow.boxes, tree['offsets'])

    @property
    def offsets(self):
        """ The offset of a box is the number of wires to its left. """
        return list(self._offsets)

    @property
    def layers(self):
        """
        A :class:`discopy.cat.Arrow` with :class:`Layer` boxes such that::

            diagram == Id(diagram.dom).then(*[
                Id(left) @ box @ Id(right)
                for left, box, right in diagram.layers])

        This is accessed using python slices::

            diagram[i:j] == Diagram(
                dom=diagram.layers[i].dom,
                cod=diagram.layers[j - 1].cod,
                boxes=diagram.boxes[i:j],
                offsets=diagram.offsets[i:j],
                layers=diagram.layers[i:j])
        """
        return self._layers

    def then(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return super().then(*others)
        other, = others
        return self.upgrade(
            Diagram(self.dom, other.cod,
                    self.boxes + other.boxes,
                    self.offsets + other.offsets,
                    layers=self.layers >> other.layers))

    def tensor(self, other=None, *rest):
        """
        Returns the horizontal composition of 'self' with a diagram 'other'.

        This method is called using the binary operator `@`:

        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> assert f0 @ f1 == f0.tensor(f1) == f0 @ Id(z) >> Id(y) @ f1

        >>> (f0 @ f1).draw(
        ...     figsize=(2, 2),
        ...     path='docs/_static/imgs/monoidal/tensor-example.png')

        .. image:: ../_static/imgs/monoidal/tensor-example.png
            :align: center

        Parameters
        ----------
        other : :class:`Diagram`

        Returns
        -------
        diagram : :class:`Diagram`
            the tensor of 'self' and 'other'.
        """
        if other is None:
            return self
        if rest:
            return self.tensor(other).tensor(*rest)
        if isinstance(other, Sum):
            return self.sum([self]).tensor(other)
        if not isinstance(other, Diagram):
            raise TypeError(messages.type_err(Diagram, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + [n + len(self.cod) for n in other.offsets]
        layers = cat.Id(dom)
        for left, box, right in self.layers:
            layers = layers >> self.layer_factory(left, box, right @ other.dom)
        for left, box, right in other.layers:
            layers = layers >> self.layer_factory(self.cod @ left, box, right)
        return self.upgrade(Diagram(dom, cod, boxes, offsets, layers=layers))

    def __matmul__(self, other):
        return self.tensor(other)

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ['dom', 'cod', 'boxes', 'offsets'])

    def __repr__(self):
        if not self.boxes:  # i.e. self is identity.
            return repr(self.id(self.dom))
        if len(self.boxes) == 1 and self.dom == self.boxes[0].dom:
            return repr(self.boxes[0])  # i.e. self is a generator.
        return "Diagram(dom={}, cod={}, boxes={}, offsets={})".format(
            repr(self.dom), repr(self.cod),
            repr(self.boxes), repr(self.offsets))

    def __hash__(self):
        return hash(repr(self))

    def __iter__(self):
        for left, box, right in self.layers:
            yield self.id(left) @ box @ self.id(right)

    def __str__(self):
        return ' >> '.join(map(str, self.layers)) or str(self.id(self.dom))

    def __getitem__(self, key):
        if isinstance(key, slice):
            layers = self.layers[key]
            boxes_and_offsets = tuple(zip(*(
                (box, len(left)) for left, box, _ in layers))) or ([], [])
            inputs = (layers.dom, layers.cod) + boxes_and_offsets
            return self.upgrade(Diagram(*inputs, layers=layers))
        left, box, right = self.layers[key]
        return self.id(left) @ box @ self.id(right)

    def subs(self, *args):
        return self.id(self.dom).then(*(
            self.id(left) @ box.subs(*args) @ self.id(right)
            for left, box, right in self.layers))

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: self.id(self.dom).then(*(
            self.id(left) @ box.lambdify(*symbols, **kwargs)(*xs)
            @ self.id(right) for left, box, right in self.layers))

    @staticmethod
    def swap(left, right, ar_factory=None, swap_factory=None):
        """
        Returns a diagram that swaps the left with the right wires.

        Parameters
        ----------
        left : monoidal.Ty
            left hand-side of the domain.
        right : monoidal.Ty
            right hand-side of the domain.

        Returns
        -------
        diagram : monoidal.Diagram
            with :code:`diagram.dom == left @ right`
        """
        ar_factory = ar_factory or Diagram
        swap_factory = swap_factory or Swap
        if not left:
            return ar_factory.id(right)
        if len(left) == 1:
            boxes = [
                swap_factory(left, right[i: i + 1])
                for i, _ in enumerate(right)]
            offsets = range(len(right))
            return ar_factory(left @ right, right @ left, boxes, offsets)
        return ar_factory.id(left[:1]) @ ar_factory.swap(left[1:], right)\
            >> ar_factory.swap(left[:1], right) @ ar_factory.id(left[1:])

    @staticmethod
    def permutation(perm, dom=None, ar_factory=None):
        """
        Returns the diagram that encodes a permutation of wires.

        Parameters
        ----------
        perm : list of int
            such that :code:`i` goes to :code:`perm[i]`
        dom : monoidal.Ty, optional
            of the same length as :code:`perm`,
            default is :code:`PRO(len(perm))`.

        Returns
        -------
        diagram : monoidal.Diagram
        """
        ar_factory = ar_factory or Diagram
        if set(range(len(perm))) != set(perm):
            raise ValueError("Input should be a permutation of range(n).")
        if dom is None:
            dom = PRO(len(perm))
        if len(dom) != len(perm):
            raise ValueError(
                "Domain and permutation should have the same length.")
        diagram = ar_factory.id(dom)
        for i in range(len(dom)):
            j = perm.index(i)
            diagram = diagram >> ar_factory.id(diagram.cod[:i])\
                @ ar_factory.swap(diagram.cod[i:j], diagram.cod[j:j + 1])\
                @ ar_factory.id(diagram.cod[j + 1:])
            perm = perm[:i] + [i] + perm[i:j] + perm[j + 1:]
        return diagram

    def permute(self, *perm):
        """
        Returns :code:`self >> self.permutation(perm, self.dom)`.

        Parameters
        ----------
        perm : list of int
            such that :code:`i` goes to :code:`perm[i]`

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Id(x @ y @ z).permute(2, 1, 0).cod == z @ y @ x
        """
        return self >> self.permutation(list(perm), self.dom)

    @staticmethod
    def subclass(ar_factory):
        """ Decorator for subclasses of Diagram. """
        def upgrade(old):
            ob_upgrade = type(ar_factory.id().dom).upgrade  # Is this Yoneda?
            dom, cod = ob_upgrade(old.dom), ob_upgrade(old.cod)
            return ar_factory(dom, cod, old.boxes, old.offsets, old.layers)
        ar_factory.upgrade = staticmethod(upgrade)
        return ar_factory

    def open_bubbles(self):
        """
        Called when drawing bubbles. Replace each bubble by::

            open_bubble\\
                >> Id(left) @ open_bubbles(bubble.inside) @ Id(right)\\
                >> close_bubble

        for :code:`left = Ty(bubble.drawing_name)` and :code:`right = Ty("")`.
        :meth:`Diagram.downgrade` gets called in the process.
        """
        if not any(isinstance(box, Bubble) for box in self.boxes):
            return self.downgrade()

        class OpenBubbles(Functor):
            def __call__(self, diagram):
                diagram = diagram.downgrade()
                if isinstance(diagram, Bubble):
                    obj = Ob(diagram.drawing_name)
                    obj.draw_as_box = True
                    left, right = Ty(obj), Ty("")
                    open_bubble = Box(
                        "open_bubble",
                        diagram.dom, left @ diagram.inside.dom @ right)
                    close_bubble = Box(
                        "_close",
                        left @ diagram.inside.cod @ right, diagram.cod)
                    open_bubble.draw_as_wires = True
                    close_bubble.draw_as_wires = True
                    # Wires can go straight only if types have the same length.
                    if len(diagram.dom) == len(diagram.inside.dom):
                        open_bubble.bubble_opening = True
                    if len(diagram.cod) == len(diagram.inside.cod):
                        close_bubble.bubble_closing = True
                    return open_bubble\
                        >> Id(left) @ self(diagram.inside) @ Id(right)\
                        >> close_bubble
                return super().__call__(diagram)
        return OpenBubbles(lambda x: x, lambda f: f)(self)

    draw = drawing.draw
    to_gif = drawing.to_gif
    interchange = rewriting.interchange
    normalize = rewriting.normalize
    normal_form = rewriting.normal_form
    foliate = rewriting.foliate
    flatten = rewriting.flatten
    foliation = rewriting.foliation
    depth = rewriting.depth
    width = rewriting.width
    layer_factory = Layer


class Id(cat.Id, Diagram):
    """ Implements the identity diagram of a given type.

    >>> s, t = Ty('x', 'y'), Ty('z', 'w')
    >>> f = Box('f', s, t)
    >>> assert f >> Id(t) == f == Id(s) >> f
    """
    def __init__(self, dom=Ty()):
        cat.Id.__init__(self, dom)
        Diagram.__init__(self, dom, dom, [], [], layers=cat.Id(dom))

    from_tree = Diagram.from_tree


Diagram.id = Id


class Box(cat.Box, Diagram):
    """
    A box is a diagram with :code:`boxes==[self]` and :code:`offsets==[0]`.

    Parameters
    ----------
    name : any
        Name of the box.
    dom : :class:`discopy.monoidal.Ty`
        Domain of the box.
    cod : :class:`discopy.monoidal.Ty`
        Codomain of the box.
    data : any, optional
        Extra data in the box.

    Other parameters
    ----------------

    draw_as_spider : bool, optional
        Whether to draw the box as a spider.
    draw_as_wires : bool, optional
        Whether to draw the box as wires, e.g. :class:`discopy.monoidal.Swap`.
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
    def downgrade(self):
        """ Downcasting to :class:`discopy.monoidal.Box`. """
        box = Box.__new__(Box)
        for attr, value in self.__dict__.items():
            setattr(box, attr, value)
        dom, cod = self.dom.downgrade(), self.cod.downgrade()
        box._dom, box._cod, box._boxes = dom, cod, [box]
        layer = Layer(box._dom[0:0], box, box._dom[0:0])
        box._layers = cat.Arrow(dom, cod, [layer], _scan=False)
        return box

    def __init__(self, name, dom, cod, **params):
        cat.Box.__init__(self, name, dom, cod, **params)
        layer = self.layer_factory(dom[0:0], self, dom[0:0])
        layers = cat.Arrow(dom, cod, [layer], _scan=False)
        Diagram.__init__(self, dom, cod, [self], [0], layers=layers)
        for attr, value in params.items():
            if attr in drawing.ATTRIBUTES:
                setattr(self, attr, value)

    def __eq__(self, other):
        if isinstance(other, Box):
            return cat.Box.__eq__(self, other)
        if isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self\
                and (other.dom, other.cod) == (self.dom, self.cod)
        return False

    def __hash__(self):
        return hash(repr(self))


class BinaryBoxConstructor:
    """ Box constructor with left and right as input. """
    def __init__(self, left, right):
        self.left, self.right = left, right

    def to_tree(self):
        left, right = self.left.to_tree(), self.right.to_tree()
        return dict(Box.to_tree(self), left=left, right=right)

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['left'], tree['right'])))


class Swap(BinaryBoxConstructor, Box):
    """
    Implements the symmetry of atomic types.

    Parameters
    ----------
    left : monoidal.Ty
        of length 1.
    right : monoidal.Ty
        of length 1.
    """
    def __init__(self, left, right):
        if len(left) != 1 or len(right) != 1:
            raise ValueError(messages.swap_vs_swaps(left, right))
        name, dom, cod =\
            "Swap({}, {})".format(left, right), left @ right, right @ left
        BinaryBoxConstructor.__init__(self, left, right)
        Box.__init__(self, name, dom, cod)
        self.draw_as_wires = True

    def __repr__(self):
        return "Swap({}, {})".format(repr(self.left), repr(self.right))

    def dagger(self):
        return type(self)(self.right, self.left)


class Sum(cat.Sum, Box):
    """ Sum of monoidal diagrams. """
    @staticmethod
    def upgrade(old):
        if not isinstance(old, cat.Sum):
            raise TypeError(messages.type_err(cat.Sum, old))
        return Sum(old.terms, old.dom, old.cod)

    def tensor(self, *others):
        if len(others) != 1:
            return super().tensor(*others)
        other = others[0] if isinstance(others[0], Sum) else Sum(others)
        unit = Sum([], self.dom @ other.dom, self.cod @ other.cod)
        terms = [f.tensor(g) for f in self.terms for g in other.terms]
        return self.upgrade(sum(terms, unit))

    def draw(self, **params):
        """ Drawing a sum as an equation with :code:`symbol='+'`. """
        return drawing.equation(*self.terms, symbol='+', **params)


class Bubble(cat.Bubble, Box):
    """
    Bubble in a monoidal diagram, i.e. a unary operator on homsets.

    Parameters
    ----------
    inside : discopy.monoidal.Diagram
        The diagram inside the bubble.
    dom : discopy.monoidal.Ty, optional
        The domain of the bubble, default is :code:`inside.dom`.
    cod : discopy.monoidal.Ty, optional
        The codomain of the bubble, default is :code:`inside.cod`.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x, y ** 3), Box('g', y, y @ y)
    >>> d = (f.bubble(dom=x @ x, cod=y) >> g).bubble()
    >>> d.draw(path='docs/_static/imgs/monoidal/bubble-example.png')

    .. image:: ../_static/imgs/monoidal/bubble-example.png
        :align: center
    """
    def __init__(self, inside, dom=None, cod=None, **params):
        self.drawing_name = params.get("drawing_name", "")
        cat.Bubble.__init__(self, inside, dom, cod)
        Box.__init__(self, self._name, self.dom, self.cod, data=self.data)

    def downgrade(self):
        """ Downcasting to :class:`discopy.monoidal.Bubble`. """
        result = Bubble(self.inside.downgrade(), Ty(*self.dom), Ty(*self.cod))
        result.drawing_name = self.drawing_name
        return result


Diagram.sum = Sum
Diagram.bubble_factory = Bubble


class Functor(cat.Functor):
    """
    Implements a monoidal functor given its image on objects and arrows.
    One may define monoidal functors into custom categories by overriding
    the defaults ob_factory=Ty and ar_factory=Diagram.

    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y, data=[0.1]), Box('f1', z, w, data=[1.1])
    >>> F = Functor({x: z, y: w, z: x, w: y}, {f0: f1, f1: f0})
    >>> assert F(f0) == f1 and F(f1) == f0
    >>> assert F(F(f0)) == f0
    >>> assert F(f0 @ f1) == f1 @ f0
    >>> assert F(f0 >> f0[::-1]) == f1 >> f1[::-1]
    >>> source, target = f0 >> f0[::-1], F(f0 >> f0[::-1])
    >>> drawing.equation(
    ...     source, target, symbol='$\\\\mapsto$', figsize=(4, 2),
    ...     path='docs/_static/imgs/monoidal/functor-example.png')

    .. image:: ../_static/imgs/monoidal/functor-example.png
        :align: center
    """
    def __init__(self, ob, ar, ob_factory=None, ar_factory=None):
        if ob_factory is None:
            ob_factory = Ty
        if ar_factory is None:
            ar_factory = Diagram
        super().__init__(ob, ar, ob_factory=ob_factory, ar_factory=ar_factory)

    def __call__(self, diagram):
        if isinstance(diagram, (Sum, Bubble)):
            super().__call__(diagram)
        if isinstance(diagram, Ty):
            return self.ob_factory().tensor(*[
                self.ob[type(diagram)(x)] for x in diagram])
        if isinstance(diagram, Swap):
            return self.ar_factory.swap(
                self(diagram.left), self(diagram.right))
        if isinstance(diagram, Box):
            return super().__call__(diagram)
        if isinstance(diagram, Diagram):
            scan, result = diagram.dom, self.ar_factory.id(self(diagram.dom))
            for box, off in zip(diagram.boxes, diagram.offsets):
                id_l = self.ar_factory.id(self(scan[:off]))
                id_r = self.ar_factory.id(self(scan[off + len(box.dom):]))
                result = result >> id_l @ self(box) @ id_r
                scan = scan[:off] @ box.cod @ scan[off + len(box.dom):]
            return result
        raise TypeError(messages.type_err(Diagram, diagram))
