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

We can check the Eckerman-Hilton argument, up to interchanger.

>>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
>>> assert s0 @ s1 == s0 >> s1 == (s1 @ s0).interchange(0, 1)
>>> assert s1 @ s0 == s1 >> s0 == (s0 @ s1).interchange(0, 1)
"""

from discopy import cat, messages, drawing
from discopy.cat import Ob, Functor, Quiver, AxiomError


class Ty(Ob):
    """
    Implements a type as a list of :class:`discopy.cat.Ob`, used as domain and
    codomain for :class:`moncat.Diagram`.
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

    def tensor(self, other):
        """
        Returns the tensor of two types, i.e. the concatenation of their lists
        of objects. This is called with the binary operator `@`.

        >>> Ty('x') @ Ty('y', 'z')
        Ty('x', 'y', 'z')

        Parameters
        ----------
        other : moncat.Ty

        Returns
        -------
        t : moncat.Ty
            such that :code:`t.objects == self.objects + other.objects`.

        Note
        ----
        We can take the sum of a list of type, specifying the unit `Ty()`.

        >>> sum([Ty('x'), Ty('y'), Ty('z')], Ty())
        Ty('x', 'y', 'z')

        We can take the exponent of a type by any natural number.

        >>> Ty('x') ** 3
        Ty('x', 'x', 'x')

        """
        return Ty(*(self.objects + other.objects))

    def __init__(self, *objects):
        self._objects = tuple(
            x if isinstance(x, Ob) else Ob(x) for x in objects)
        super().__init__(str(self))

    def __eq__(self, other):
        if not isinstance(other, Ty):
            return False
        return self.objects == other.objects

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return "Ty({})".format(', '.join(repr(x.name) for x in self.objects))

    def __str__(self):
        return ' @ '.join(map(str, self)) or 'Ty()'

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Ty(*self.objects[key])
        return self.objects[key]

    def __matmul__(self, other):
        return self.tensor(other)

    def __add__(self, other):
        return self.tensor(other)

    def __pow__(self, n_times):
        if not isinstance(n_times, int):
            raise TypeError(messages.type_err(int, n_times))
        return sum(n_times * (self, ), type(self)())


class PRO(Ty):
    """ Implements the objects of a PRO, i.e. a non-symmetric PROP.
    Wraps a natural number n into a unary type Ty(1, ..., 1) of length n.

    >>> PRO(1) @ PRO(1)
    PRO(2)
    >>> assert PRO(3) == Ty(1, 1, 1)
    >>> assert PRO(1) == PRO(Ob(1))
    """
    def __init__(self, n=0):
        if isinstance(n, PRO):
            n = len(n)
        if isinstance(n, Ob):
            n = n.name
        super().__init__(*(n * [1]))

    def tensor(self, other):
        if not isinstance(other, PRO):
            if isinstance(other, Ty):
                return super().tensor(other)
            raise TypeError(messages.type_err(PRO, other))
        return type(self)(len(self) + len(other))

    def __repr__(self):
        return "PRO({})".format(len(self))

    def __str__(self):
        return repr(len(self))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(len(super().__getitem__(key)))
        return super().__getitem__(key)


class Layer(cat.Box):
    """
    Layer of a diagram, i.e. a box with wires to the left and right.

    Parameters
    ----------
    left : moncat.Ty
        Left wires.
    box : moncat.Box
        Middle box.
    right : moncat.Ty
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
        name = (left, box, right)
        dom, cod = left @ box.dom @ right, left @ box.cod @ right
        super().__init__(name, dom, cod)

    def __iter__(self):
        yield self._left
        yield self._box
        yield self._right

    def __repr__(self):
        return "Layer({}, {}, {})".format(
            *map(repr, (self._left, self._box, self._right)))

    def __str__(self):
        return ("{} @ ".format(cat.Id(self._left)) if self._left else "")\
            + str(self._box)\
            + (" @ {}".format(cat.Id(self._right)) if self._right else "")

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return Layer(self._left, self._box[::-1], self._right)
        return super().__getitem__(key)


class Diagram(cat.Arrow):
    """
    Defines a diagram given dom, cod, a list of boxes and offsets.

    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1, g = Box('f0', x, y), Box('f1', z, w), Box('g', y @ w, y)
    >>> d = Diagram(x @ z, y, [f0, f1, g], [0, 1, 0])
    >>> assert d == f0 @ f1 >> g

    Parameters
    ----------
    dom : :class:`Ty`
        Domain of the diagram.
    cod : :class:`Ty`
        Codomain of the diagram.
    boxes : list of :class:`Diagram`
        Boxes of the diagram.
    offsets : list of int
        Offsets of each box in the diagram.
    layers : list of :class:`Layer`, optional
        Layers of the diagram, computed from boxes and offsets if :code:`None`.

    Raises
    ------
    :class:`AxiomError`
        Whenever the boxes do not compose.
    """
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
                layers = layers >> Layer(left, box, right)
            layers = layers >> cat.Id(cod)
        self._layers, self._offsets = layers, tuple(offsets)
        super().__init__(dom, cod, boxes, _scan=False)

    @staticmethod
    def id(x):
        return Id(x)

    @property
    def offsets(self):
        """
        The offset of a box is the number of wires to its left.
        """
        return list(self._offsets)

    @property
    def layers(self):
        """
        A :class:`discopy.cat.Arrow` with :class:`Layer` boxes such that::

            diagram == Id(diagram.dom).compose(*[
                Id(left) @ box Id(right)
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

    def then(self, other):
        return Diagram(self.dom, other.cod,
                       self.boxes + other.boxes,
                       self.offsets + other.offsets,
                       layers=self.layers >> other.layers)

    def tensor(self, other):
        """
        Returns the horizontal composition of 'self' with a diagram 'other'.

        This method is called using the binary operator `@`:

        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> assert f0 @ f1 == f0.tensor(f1) == f0 @ Id(z) >> Id(y) @ f1

        Parameters
        ----------
        other : :class:`Diagram`

        Returns
        -------
        diagram : :class:`Diagram`
            the tensor of 'self' and 'other'.
        """
        if not isinstance(other, Diagram):
            raise TypeError(messages.type_err(Diagram, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + [n + len(self.cod) for n in other.offsets]
        layers = cat.Id(dom)
        for left, box, right in self.layers:
            layers = layers >> Layer(left, box, right @ other.dom)
        for left, box, right in other.layers:
            layers = layers >> Layer(self.cod @ left, box, right)
        return Diagram(dom, cod, boxes, offsets, layers=layers)

    def __matmul__(self, other):
        return self.tensor(other)

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return all(self.__getattribute__(attr) == other.__getattribute__(attr)
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
        result = ' >> '.join(map(str, self.layers)) or str(self.id(self.dom))
        if len(result) > 74:
            result = result.replace(' >>', '\\\n  >>')
        return result

    def __getitem__(self, key):
        if isinstance(key, slice):
            layers = self.layers[key]
            boxes_and_offsets = tuple(zip(*(
                (box, len(left)) for left, box, _ in layers))) or ([], [])
            inputs = (layers.dom, layers.cod) + boxes_and_offsets
            return Diagram(*inputs, layers=layers)
        left, box, right = self.layers[key]
        return self.id(left) @ box @ self.id(right)

    def interchange(self, i, j, left=False):
        """
        Returns a new diagram with boxes i and j interchanged.

        Gets called recursively whenever :code:`i < j + 1 or j < i - 1`.

        Parameters
        ----------
        i : int
            Index of the box to interchange.
        j : int
            Index of the new position for the box.
        left : bool, optional
            Whether to apply left interchangers.

        Notes
        -----
        By default, we apply only right exchange moves::

            top >> Id(left @ box1.dom @ mid) @ box0 @ Id(right)
                >> Id(left) @ box1 @ Id(mid @ box0.cod @ right) >> bottom

        gets rewritten to::

            top >> Id(left) @ box1 @ Id(mid @ box0.dom @ right)
                >> Id(left @ box1.cod @ mid) @ box0 @ Id(right) >> bottom
        """
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
        left0, box0, right0 = self.layers[i]
        left1, box1, right1 = self.layers[j]
        # By default, we check if box0 is to the right first, then to the left.
        if left and off1 >= off0 + len(box0.cod):  # box0 left of box1
            off1 = off1 - len(box0.cod) + len(box0.dom)
            middle = left1[len(left0 @ box0.cod):]
            layer0 = Layer(left0, box0, middle @ box1.cod @ right1)
            layer1 = Layer(left0 @ box0.dom @ middle, box1, right1)
        elif off0 >= off1 + len(box1.dom):  # box0 right of box1
            off0 = off0 - len(box1.dom) + len(box1.cod)
            middle = left0[len(left1 @ box1.dom):]
            layer0 = Layer(left1 @ box1.cod @ middle, box0, right0)
            layer1 = Layer(left1, box1, middle @ box0.dom @ right0)
        elif off1 >= off0 + len(box0.cod):  # box0 left of box1
            off1 = off1 - len(box0.cod) + len(box0.dom)
            middle = left1[len(left0 @ box0.cod):]
            layer0 = Layer(left0, box0, middle @ box1.cod @ right1)
            layer1 = Layer(left0 @ box0.dom @ middle, box1, right1)
        else:
            raise InterchangerError(box0, box1)
        boxes = self.boxes[:i] + [box1, box0] + self.boxes[i + 2:]
        offsets = self.offsets[:i] + [off1, off0] + self.offsets[i + 2:]
        layers = self.layers[:i] >> layer1 >> layer0 >> self.layers[i + 2:]
        return Diagram(self.dom, self.cod, boxes, offsets, layers=layers)

    def normalize(self, left=False):
        """
        Implements normalisation of connected diagrams, see arXiv:1804.07832.

        Parameters
        ----------
        left : bool, optional
            Passed to :meth:`Diagram.interchange`.

        Yields
        ------
        diagram : :class:`Diagram`
            Rewrite steps.

        Examples
        --------

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

    def normal_form(self, normalize=None, **params):
        """
        Returns the normal form of a diagram.

        Parameters
        ----------
        normalize : iterable of :class:`Diagram`, optional
            Generator that yields rewrite steps, default is
            :meth:`Diagram.normalize`.

        params : any, optional
            Passed to :code:`normalize`.

        Raises
        ------
        NotImplementedError
            Whenever :code:`normalize` yields the same rewrite steps twice.
        """
        diagram, cache = self, set()
        for _diagram in (normalize or Diagram.normalize)(diagram, **params):
            if _diagram in cache:
                raise NotImplementedError(messages.is_not_connected(self))
            diagram = _diagram
            cache.add(diagram)
        return diagram

    def foliate(self, yield_slices=False):
        """
        Generator yielding the interchanger steps in the foliation of self.

        Yields
        ------
        diagram : :class:`Diagram`
            Rewrite steps of the foliation.

        Parameters
        ----------
        yield_slices : bool, optional
            Yield the list of slices of self as last output,
            used in :meth:`Diagram.foliation`.

        Examples
        --------

        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> d = (f0 @ Id(y) >> f0.dagger() @ f1) @ (f0 >> f1)
        >>> *_, slices = d.foliate(yield_slices=True)
        >>> print(slices[0])
        f0 @ Id(y @ x) >> Id(y) @ f1 @ Id(x) >> Id(y @ x) @ f0
        >>> print(slices[1])
        f0[::-1] @ Id(x @ y) >> Id(x @ x) @ f1
        >>> ket = Box('ket', Ty(), x)
        >>> scalar = Box('scalar', Ty(), Ty())
        >>> kets = scalar @ ket @ scalar @ ket
        >>> a = kets.foliate()
        >>> assert next(a) == kets
        """
        def is_right_of(last, diagram):
            off0, off1 = diagram.offsets[last], diagram.offsets[last + 1]
            box0, box1 = diagram.boxes[last], diagram.boxes[last + 1]
            if off1 >= off0 + len(box0.cod):  # box1 right of box0
                return True
            if off0 >= off1 + len(box1.dom):  # box1 left of box0
                return False
            return None

        def move_in_slice(first, last, k, diagram):
            result = diagram
            try:
                if not k == last + 1:
                    result = diagram.interchange(k, last + 1)
                right_of_last = is_right_of(last, result)
                if right_of_last is None:
                    return None
                if right_of_last:
                    return result
                result = result.interchange(last + 1, last)
                if last == first:
                    return result
                return move_in_slice(first, last - 1, last, result)
            except InterchangerError:
                return None

        start, diagram = 0, self
        if yield_slices:
            slices = []
        while start < len(diagram):
            last = start
            k = last + 1
            while k < len(diagram):
                result = move_in_slice(start, last, k, diagram)
                k += 1
                if result is None:
                    pass
                else:
                    diagram = result
                    last += 1
                    yield diagram
            if yield_slices:
                slices += [diagram[start: last + 1]]
            start = last + 1
        if yield_slices:
            yield slices

    def flatten(self):
        """
        Takes a diagram of diagrams and returns a diagram.

        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> g = Box('g', x @ y, y)
        >>> d = (Id(y) @ f0 @ Id(x) >> f0.dagger() @ Id(y) @ f0 >>\\
        ...      g @ f1 >> f1 @ Id(x)).normal_form()
        >>> assert d.foliation().flatten().normal_form() == d
        >>> assert d.foliation().dagger().flatten()\\
        ...     == d.foliation().flatten().dagger()
        """
        return Functor(Quiver(lambda x: x), Quiver(lambda f: f))(self)

    def foliation(self):
        """
        Returns a diagram with normal_form diagrams of depth 1 as boxes
        such that its flattening gives the original diagram back.

        >>> x, y = Ty('x'), Ty('y')
        >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
        >>> d = f0 @ Id(y) >> f0.dagger() @ f1
        >>> assert d.foliation().boxes[0] == f0 @ f1
        >>> assert d.foliation().flatten().normal_form() == d
        >>> assert d.foliation().flatten()\\
        ...     == d[::-1].foliation()[::-1].flatten()\\
        ...     == d[::-1].foliation().flatten()[::-1]
        >>> assert d.foliation().flatten().foliation() == d.foliation()
        >>> g = Box('g', x @ x, x @ y)
        >>> diagram = (d >> g >> d) @ (d >> g >> d)
        >>> slices = diagram.foliation()
        >>> assert slices.boxes[0] == f0 @ f1 @ f0 @ f1
        >>> *_, last_diagram = diagram.foliate()
        >>> assert last_diagram == slices.flatten()
        """
        *_, slices = self.foliate(yield_slices=True)
        return Diagram(self.dom, self.cod, slices, len(slices) * [0])

    def depth(self):
        """
        Computes the depth of a diagram by foliating it

        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', y, x)
        >>> assert Id(x @ y).depth() == 0
        >>> assert f.depth() == 1
        >>> assert (f @ g).depth() == 1
        >>> assert (f >> g).depth() == 2
        """
        *_, slices = self.foliate(yield_slices=True)
        return len(slices)

    def width(self):
        """
        Computes the width of a diagram,
        i.e. the maximum number of parallel wires.

        >>> x = Ty('x')
        >>> f = Box('f', x, x ** 4)
        >>> assert (f >> f.dagger()).width() == 4
        >>> assert (f @ Id(x ** 2) >> Id(x ** 2) @ f.dagger()).width() == 6
        """
        scan = self.dom
        width = len(scan)
        for box, off in zip(self.boxes, self.offsets):
            scan = scan[: off] + box.cod + scan[off + len(box.dom):]
            width = max(width, len(scan))
        return width

    def draw(self, **params):
        """
        Draws a diagram using networkx and matplotlib.

        Parameters
        ----------
        draw_as_nodes : bool, optional
            Whether to draw boxes as nodes, default is :code:`False`.
        color : string, optional
            Color of the box or node, default is white (:code:`'#ffffff'`) for
            boxes and red (:code:`'#ff0000'`) for nodes.
        textpad : float, optional
            Padding between text and wires, default is :code:`0.1`.
        draw_types : bool, optional
            Whether to draw type labels, default is :code:`False`.
        draw_box_labels : bool, optional
            Whether to draw box labels, default is :code:`True`.
        aspect : string, optional
            Aspect ratio, one of :code:`['equal', 'auto']`.
        margins : tuple, optional
            Margins, default is :code:`(0.05, 0.05)`.
        fontsize : int, optional
            Font size for the boxes, default is :code:`12`.
        fontsize_types : int, optional
            Font size for the types, default is :code:`12`.
        figsize : tuple, optional
            Figure size.
        path : str, optional
            Where to save the image, if `None` we call :code:`plt.show()`.
        """
        return drawing.draw(self, **params)

    def to_gif(self, *diagrams, **params):
        """
        Builds a gif with the normalisation steps.

        Parameters
        ----------
        diagrams : :class:`Diagram`, optional
            Sequence of diagrams to draw.
        path : str
            Where to save the image, if :code:`None` a gif gets created.
        timestep : int, optional
            Time step in milliseconds, default is :code:`500`.
        loop : bool, optional
            Whether to loop, default is :code:`False`
        params : any, optional
            Passed to :meth:`Diagram.draw`.
        """
        return drawing.to_gif(self, *diagrams, **params)


def spiral(n_cups, _type=Ty('x')):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    unit, counit = Box('unit', Ty(), _type), Box('counit', _type, Ty())
    cup, cap = Box('cup', _type @ _type, Ty()), Box('cap', Ty(), _type @ _type)
    result = unit
    for i in range(n_cups):
        result = result >> Id(_type ** i) @ cap @ Id(_type ** (i + 1))
    result = result >> Id(_type ** n_cups) @ counit @ Id(_type ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(_type ** (n_cups - i - 1)) @ cup @ Id(_type ** (n_cups - i - 1))
    return result


class InterchangerError(AxiomError):
    """
    This is raised when we try to interchange conected boxes.
    """
    def __init__(self, box0, box1):
        super().__init__("Boxes {} and {} do not commute.".format(box0, box1))


class Id(Diagram):
    """ Implements the identity diagram of a given type.

    >>> s, t = Ty('x', 'y'), Ty('z', 'w')
    >>> f = Box('f', s, t)
    >>> assert f >> Id(t) == f == Id(s) >> f
    """
    def __init__(self, x):
        super().__init__(x, x, [], [], layers=cat.Id(x))

    def __repr__(self):
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        return "Id({})".format(str(self.dom))


class Box(cat.Box, Diagram):
    """
    Implements a box as a diagram with :code:`boxes=[self], offsets=[0]`.

    >>> f = Box('f', Ty('x', 'y'), Ty('z'))
    >>> assert Id(Ty('x', 'y')) >> f == f == f >> Id(Ty('z'))
    >>> assert Id(Ty()) @ f == f == f @ Id(Ty())
    >>> assert f == f[::-1][::-1]
    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        cat.Box.__init__(self, name, dom, cod, data=data, _dagger=_dagger)
        layers = cat.Arrow(dom, cod, [Layer(Ty(), self, Ty())], _scan=False)
        Diagram.__init__(self, dom, cod, [self], [0], layers=layers)

    def __eq__(self, other):
        if isinstance(other, Box):
            return cat.Box.__eq__(self, other)
        if isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self\
                and (other.dom, other.cod) == (self.dom, self.cod)
        return False

    def __hash__(self):
        return hash(repr(self))


class Functor(cat.Functor):
    """
    Implements a monoidal functor given its image on objects and arrows.
    One may define monoidal functors into custom categories by overriding
    the defaults ob_cls=Ty and ar_cls=Diagram.

    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y, data=[0.1]), Box('f1', z, w, data=[1.1])
    >>> F = Functor({x: z, y: w, z: x, w: y}, {f0: f1, f1: f0})
    >>> assert F(f0) == f1 and F(f1) == f0
    >>> assert F(F(f0)) == f0
    >>> assert F(f0 @ f1) == f1 @ f0
    >>> assert F(f0 >> f0[::-1]) == f1 >> f1[::-1]
    """
    def __init__(self, ob, ar, ob_cls=None, ar_cls=None):
        if ob_cls is None:
            ob_cls = Ty
        if ar_cls is None:
            ar_cls = Diagram
        super().__init__(ob, ar, ob_cls=ob_cls, ar_cls=ar_cls)

    def __call__(self, diagram):
        if isinstance(diagram, Ty):
            return sum([self.ob[type(diagram)(x)] for x in diagram],
                       self.ob_cls())  # the empty type is the unit for sum.
        if isinstance(diagram, Box):
            return super().__call__(diagram)
        if isinstance(diagram, Diagram):
            scan, result = diagram.dom, self.ar_cls.id(self(diagram.dom))
            for box, off in zip(diagram.boxes, diagram.offsets):
                id_l = self.ar_cls.id(self(scan[:off]))
                id_r = self.ar_cls.id(self(scan[off + len(box.dom):]))
                result = result >> id_l @ self(box) @ id_r
                scan = scan[:off] + box.cod + scan[off + len(box.dom):]
            return result
        raise TypeError(messages.type_err(Diagram, diagram))
