# -*- coding: utf-8 -*-

"""
Implements the free rigid monoidal category.

The objects are given by the free pregroup, the arrows by planar diagrams.

>>> unit, s, n = Ty(), Ty('s'), Ty('n')
>>> t = n.r @ s @ n.l
>>> assert t @ unit == t == unit @ t
>>> assert t.l.r == t == t.r.l
>>> left_snake, right_snake = Id(n.r).transpose(left=True), Id(n.l).transpose()
>>> assert left_snake.normal_form() == Id(n) == right_snake.normal_form()
>>> from discopy import drawing
>>> drawing.equation(
...     left_snake, Id(n), right_snake, figsize=(4, 2),
...     path='docs/_static/imgs/rigid/snake-equation.png')

.. image:: ../_static/imgs/rigid/snake-equation.png
    :align: center
"""

from discopy import cat, monoidal, messages, rewriting
from discopy.cat import AxiomError


class Ob(cat.Ob):
    """
    Implements simple pregroup types: basic types and their iterated adjoints.

    >>> a = Ob('a')
    >>> assert a.l.r == a.r.l == a and a != a.l.l != a.r.r
    """
    @property
    def z(self):
        """ Winding number """
        return self._z

    @property
    def l(self):
        """ Left adjoint """
        return Ob(self.name, self.z - 1)

    @property
    def r(self):
        """ Right adjoint """
        return Ob(self.name, self.z + 1)

    def __init__(self, name, z=0):
        if not isinstance(z, int):
            raise TypeError(messages.type_err(int, z))
        self._z = z
        super().__init__(name)

    def __eq__(self, other):
        if not isinstance(other, Ob):
            if isinstance(other, cat.Ob):
                return self.z == 0 and self.name == other.name
            return False
        return (self.name, self.z) == (other.name, other.z)

    def __hash__(self):
        return hash(self.name if not self.z else (self.name, self.z))

    def __repr__(self):
        return "Ob({}{})".format(
            repr(self.name), ", z=" + repr(self.z) if self.z else '')

    def __str__(self):
        return str(self.name) + (
            - self.z * '.l' if self.z < 0 else self.z * '.r')

    def to_tree(self):
        tree = super().to_tree()
        if self.z:
            tree['z'] = self.z
        return tree

    @classmethod
    def from_tree(cls, tree):
        name, z = tree['name'], tree.get('z', 0)
        return cls(name=name, z=z)


class Ty(monoidal.Ty, Ob):
    """ Implements pregroup types as lists of simple types.

    >>> s, n = Ty('s'), Ty('n')
    >>> assert n.l.r == n == n.r.l
    >>> assert (s @ n).l == n.l @ s.l and (s @ n).r == n.r @ s.r
    """
    @staticmethod
    def upgrade(old):
        return Ty(*old.objects)

    @property
    def l(self):
        return Ty(*[x.l for x in self.objects[::-1]])

    @property
    def r(self):
        return Ty(*[x.r for x in self.objects[::-1]])

    @property
    def z(self):
        if len(self) != 1:
            raise TypeError(messages.no_winding_number_for_complex_types())
        return self[0].z

    def __init__(self, *t):
        t = [x if isinstance(x, Ob)
             else Ob(x.name) if isinstance(x, cat.Ob)
             else Ob(x) for x in t]
        monoidal.Ty.__init__(self, *t)
        Ob.__init__(self, str(self))

    def __repr__(self):
        return "Ty({})".format(', '.join(
            repr(x if x.z else x.name) for x in self.objects))

    def __lshift__(self, other):
        return self @ other.l

    def __rshift__(self, other):
        return self.r @ other


class PRO(monoidal.PRO, Ty):
    """
    Objects of the free rigid monoidal category generated by 1.
    """
    @staticmethod
    def upgrade(old):
        return PRO(len(monoidal.PRO.upgrade(old)))

    @property
    def l(self):
        """
        >>> assert PRO(2).l == PRO(2)
        """
        return self

    @property
    def r(self):
        return self


class Layer(monoidal.Layer):
    @staticmethod
    def upgrade(old):
        return Layer(old._left, old._box, old._right)

    @property
    def l(self):
        return Layer(self._right.l, self._box.l, self._left.l)

    @property
    def r(self):
        return Layer(self._right.r, self._box.r, self._left.r)


@monoidal.Diagram.subclass
class Diagram(monoidal.Diagram):
    """ Implements diagrams in the free rigid monoidal category.

    >>> I, n, s = Ty(), Ty('n'), Ty('s')
    >>> Alice, jokes = Box('Alice', I, n), Box('jokes', I, n.r @ s)
    >>> boxes, offsets = [Alice, jokes, Cup(n, n.r)], [0, 1, 0]
    >>> d = Diagram(Alice.dom @ jokes.dom, s, boxes, offsets)
    >>> print(d)
    Alice >> Id(n) @ jokes >> Cup(n, n.r) @ Id(s)

    >>> d.draw(figsize=(3, 2),
    ...        path='docs/_static/imgs/rigid/diagram-example.png')

    .. image:: ../_static/imgs/rigid/diagram-example.png
        :align: center
    """
    @staticmethod
    def swap(left, right):
        return monoidal.Diagram.swap(
            left, right, ar_factory=Diagram, swap_factory=Swap)

    @staticmethod
    def permutation(perm, dom=None):
        if dom is None:
            dom = PRO(len(perm))
        return monoidal.Diagram.permutation(perm, dom, ar_factory=Diagram)

    def foliate(self, yield_slices=False):
        """
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> gen = (f @ Id(x) >> (f @ f)).foliate()
        >>> print(next(gen))
        f @ Id(x) >> Id(x) @ f >> f @ Id(x)
        """
        for diagram in super().foliate(yield_slices=yield_slices):
            if isinstance(diagram, cat.Arrow):
                yield self.upgrade(diagram)
            else:
                yield [self.upgrade(diagram[i]) for i in range(len(diagram))]

    @staticmethod
    def cups(left, right):
        """ Constructs nested cups witnessing adjointness of x and y.

        >>> a, b = Ty('a'), Ty('b')
        >>> assert Diagram.cups(a, a.r) == Cup(a, a.r)
        >>> assert Diagram.cups(a @ b, (a @ b).r) ==\\
        ...     Id(a) @ Cup(b, b.r) @ Id(a.r) >> Cup(a, a.r)

        >>> Diagram.cups(a @ b, (a @ b).r).draw(figsize=(3, 1),\\
        ... margins=(0.3, 0.05), path='docs/_static/imgs/rigid/cups.png')

    .. image:: ../_static/imgs/rigid/cups.png
        :align: center
        """
        return cups(left, right)

    @staticmethod
    def caps(left, right):
        """ Constructs nested cups witnessing adjointness of x and y.

        >>> a, b = Ty('a'), Ty('b')
        >>> assert Diagram.caps(a, a.l) == Cap(a, a.l)
        >>> assert Diagram.caps(a @ b, (a @ b).l) == (Cap(a, a.l)
        ...                 >> Id(a) @ Cap(b, b.l) @ Id(a.l))
        """
        return caps(left, right)

    @staticmethod
    def spiders(n_legs_in, n_legs_out, typ):
        """ Constructs spiders with compound types."""
        return spiders(n_legs_in, n_legs_out, typ)

    @staticmethod
    def fa(left, right):
        """ Forward application. """
        off = -len(right) or len(left)
        return Id(left[:off]) @ Diagram.cups(left[off:], right)

    @staticmethod
    def ba(left, right):
        """ Backward application. """
        off = len(left) or -len(right)
        return Diagram.cups(left, right[:off]) @ Id(right[off:])

    @staticmethod
    def fc(left, middle, right):
        """ Forward composition. """
        return Id(left) @ Diagram.cups(middle.l, middle) @ Id(right.l)

    @staticmethod
    def bc(left, middle, right):
        """ Backward composition. """
        return Id(left.r) @ Diagram.cups(middle, middle.r) @ Id(right)

    @staticmethod
    def fx(left, middle, right):
        """ Forward crossed composition. """
        return Id(left) @ Diagram.swap(middle.l, right.r) @ Id(middle) >>\
            Diagram.swap(left, right.r) @ Diagram.cups(middle.l, middle)

    @staticmethod
    def bx(left, middle, right):
        """ Backward crossed composition. """
        return Id(middle) @ Diagram.swap(left.l, middle.r) @ Id(right) >>\
            Diagram.cups(middle, middle.r) @ Diagram.swap(left.l, right)

    @staticmethod
    def curry(diagram, n_wires=1, left=False):
        """ Diagram currying. """
        if left:
            wires = diagram.dom[:n_wires]
            return Diagram.caps(wires.r, wires) @ Id(diagram.dom[n_wires:])\
                >> Id(wires.r) @ diagram
        wires = diagram.dom[-n_wires or len(diagram.dom):]
        return Id(diagram.dom[:-n_wires]) @ Diagram.caps(wires, wires.l)\
            >> diagram @ Id(wires.l)

    def _conjugate(self, use_left):
        layers = self.layers
        list_of_layers = []
        for layer in layers._boxes:
            layer_adj = layer.l if use_left else layer.r
            left, box, right = layer_adj
            list_of_layers += (Id(left) @ box @ Id(right)).layers.boxes

        dom = layers.dom.l if use_left else layers.dom.r
        cod = layers.cod.l if use_left else layers.cod.r
        layers_adj = type(layers)(dom, cod, list_of_layers)
        boxes_and_offsets = tuple(zip(*(
            (box, len(left)) for left, box, _ in layers_adj))) or ([], [])
        inputs = (dom, cod) + boxes_and_offsets
        return self.upgrade(Diagram(*inputs, layers=layers_adj))

    @property
    def l(self):
        return self._conjugate(use_left=True)

    @property
    def r(self):
        return self._conjugate(use_left=False)

    def dagger(self):
        d = super().dagger()
        d._layers._boxes = [Layer.upgrade(b) for b in d._layers._boxes]
        return d

    def transpose_box(self, i, left=False):
        bend_left = left
        layers = self.layers
        if bend_left:
            box_T = layers[i]._box.r.dagger().transpose(left=True)
        else:
            box_T = layers[i]._box.l.dagger().transpose(left=False)
        left, _, right = layers[i]
        layers_T = (Id(left) @ box_T @ Id(right)).layers.boxes
        list_of_layers = layers._boxes[:i] + layers_T + layers._boxes[i + 1:]
        layers = type(layers)(layers.dom, layers.cod, list_of_layers)
        boxes_and_offsets = tuple(zip(*(
            (box, len(left)) for left, box, _ in layers))) or ([], [])
        inputs = (layers.dom, layers.cod) + boxes_and_offsets
        return self.upgrade(Diagram(*inputs, layers=layers))

    def transpose(self, left=False):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> double_snake = Id(a @ b).transpose()
        >>> two_snakes = Id(b).transpose() @ Id(a).transpose()
        >>> double_snake == two_snakes
        False
        >>> *_, two_snakes_nf = monoidal.Diagram.normalize(two_snakes)
        >>> assert double_snake == two_snakes_nf
        >>> f = Box('f', a, b)

        >>> a, b = Ty('a'), Ty('b')
        >>> double_snake = Id(a @ b).transpose(left=True)
        >>> snakes = Id(b).transpose(left=True) @ Id(a).transpose(left=True)
        >>> double_snake == two_snakes
        False
        >>> *_, two_snakes_nf = monoidal.Diagram.normalize(
        ...     snakes, left=True)
        >>> assert double_snake == two_snakes_nf
        >>> f = Box('f', a, b)
        """
        if left:
            return self.id(self.cod.l) @ self.caps(self.dom, self.dom.l)\
                >> self.id(self.cod.l) @ self @ self.id(self.dom.l)\
                >> self.cups(self.cod.l, self.cod) @ self.id(self.dom.l)
        return self.caps(self.dom.r, self.dom) @ self.id(self.cod.r)\
            >> self.id(self.dom.r) @ self @ self.id(self.cod.r)\
            >> self.id(self.dom.r) @ self.cups(self.cod, self.cod.r)

    def normal_form(self, normalizer=None, **params):
        """
        Implements the normalisation of rigid monoidal categories,
        see arxiv:1601.05372, definition 2.12.
        """
        return super().normal_form(
            normalizer=normalizer or Diagram.normalize, **params)

    normalize = rewriting.snake_removal
    layer_factory = Layer

    def cup(self, x, y):
        if min(x, y) < 0 or max(x, y) >= len(self.cod):
            raise ValueError(f'Indices {x, y} are out of range.')
        x, y = min(x, y), max(x, y)
        for i in range(x, y - 1):
            t0, t1 = self.cod[i:i + 1], self.cod[i + 1:i + 2]
            self >>= Id(self.cod[:i]) @ Swap(t0, t1) @ Id(self.cod[i + 2:])
        t0, t1 = self.cod[y - 1:y], self.cod[y:y + 1]
        self >>= Id(self.cod[:y - 1]) @ Cup(t0, t1) @ Id(self.cod[y + 1:])
        return self


Sum = cat.Sum

Sum.l = property(cat.Sum.fmap(lambda d: d.l))
Sum.r = property(cat.Sum.fmap(lambda d: d.r))


class Id(monoidal.Id, Diagram):
    """ Define an identity arrow in a free rigid category

    >>> t = Ty('a', 'b', 'c')
    >>> assert Id(t) == Diagram(t, t, [], [])
    """
    def __init__(self, dom=Ty()):
        monoidal.Id.__init__(self, dom)
        Diagram.__init__(self, dom, dom, [], [], layers=cat.Id(dom))

    @property
    def l(self):
        return type(self)(self.dom.l)

    @property
    def r(self):
        return type(self)(self.dom.r)


Diagram.id = Id


class Box(monoidal.Box, Diagram):
    """ Implements generators of rigid monoidal diagrams.

    >>> a, b = Ty('a'), Ty('b')
    >>> Box('f', a, b.l @ b, data={42})
    Box('f', Ty('a'), Ty(Ob('b', z=-1), 'b'), data={42})
    """
    def __init__(self, name, dom, cod, **params):
        monoidal.Box.__init__(self, name, dom, cod, **params)
        Diagram.__init__(self, dom, cod, [self], [0], layers=self.layers)
        self._z = params.get("_z", 0)

    def __eq__(self, other):
        if isinstance(other, Box):
            return self._z == other._z and monoidal.Box.__eq__(self, other)
        if isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self\
                and (other.dom, other.cod) == (self.dom, self.cod)
        return False

    def __hash__(self):
        return hash(repr(self))

    @property
    def z(self):
        return self._z

    def dagger(self):
        return type(self)(
            name=self.name, dom=self.cod, cod=self.dom,
            data=self.data, _dagger=not self._dagger, _z=self._z)

    @property
    def l(self):
        return type(self)(
            name=self.name, dom=self.dom.l, cod=self.cod.l,
            data=self.data, _dagger=self._dagger, _z=self._z - 1)

    @property
    def r(self):
        return type(self)(
            name=self.name, dom=self.dom.r, cod=self.cod.r,
            data=self.data, _dagger=self._dagger, _z=self._z + 1)


class Swap(monoidal.Swap, Box):
    """ Implements swaps of basic types in a rigid category. """
    def __init__(self, left, right):
        monoidal.Swap.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod)


class Cup(monoidal.BinaryBoxConstructor, Box):
    """ Defines cups for simple types.

    >>> n = Ty('n')
    >>> Cup(n, n.r)
    Cup(Ty('n'), Ty(Ob('n', z=1)))

    >>> Cup(n, n.r).draw(figsize=(2,1), margins=(0.5, 0.05),\\
    ... path='docs/_static/imgs/rigid/cup.png')

    .. image:: ../_static/imgs/rigid/cup.png
        :align: center
    """
    def __init__(self, left, right):
        if not isinstance(left, Ty):
            raise TypeError(messages.type_err(Ty, left))
        if not isinstance(right, Ty):
            raise TypeError(messages.type_err(Ty, right))
        if len(left) != 1 or len(right) != 1:
            raise ValueError(messages.cup_vs_cups(left, right))
        if left.r != right and left != right.r:
            raise AxiomError(messages.are_not_adjoints(left, right))
        monoidal.BinaryBoxConstructor.__init__(self, left, right)
        Box.__init__(
            self, "Cup({}, {})".format(left, right), left @ right, Ty())
        self.draw_as_wires = True

    def dagger(self):
        return Cap(self.left, self.right)

    @property
    def l(self):
        return Cup(self.right.l, self.left.l)

    @property
    def r(self):
        return Cup(self.right.r, self.left.r)

    def __repr__(self):
        return "Cup({}, {})".format(repr(self.left), repr(self.right))


class Cap(monoidal.BinaryBoxConstructor, Box):
    """ Defines cups for simple types.

    >>> n = Ty('n')
    >>> Cap(n, n.l)
    Cap(Ty('n'), Ty(Ob('n', z=-1)))

    >>> Cap(n, n.l).draw(figsize=(2,1), margins=(0.5, 0.05),\\
    ... path='docs/_static/imgs/rigid/cap.png')

    .. image:: ../_static/imgs/rigid/cap.png
        :align: center
    """
    def __init__(self, left, right):
        if not isinstance(left, Ty):
            raise TypeError(messages.type_err(Ty, left))
        if not isinstance(right, Ty):
            raise TypeError(messages.type_err(Ty, right))
        if len(left) != 1 or len(right) != 1:
            raise ValueError(messages.cap_vs_caps(left, right))
        if left != right.r and left.r != right:
            raise AxiomError(messages.are_not_adjoints(left, right))
        monoidal.BinaryBoxConstructor.__init__(self, left, right)
        Box.__init__(
            self, "Cap({}, {})".format(left, right), Ty(), left @ right)
        self.draw_as_wires = True

    def dagger(self):
        return Cup(self.left, self.right)

    @property
    def l(self):
        return Cap(self.right.l, self.left.l)

    @property
    def r(self):
        return Cap(self.right.r, self.left.r)

    def __repr__(self):
        return "Cap({}, {})".format(repr(self.left), repr(self.right))


class Spider(Box):
    """
    Spider box.

    Parameters
    ----------
    n_legs_in, n_legs_out : int
        Number of legs in and out.
    typ : discopy.rigid.Ty
        The type of the spider, needs to be atomic.

    Examples
    --------
    >>> x = Ty('x')
    >>> spider = Spider(1, 2, x)
    >>> assert spider.dom == x and spider.cod == x @ x
    """
    def __init__(self, n_legs_in, n_legs_out, typ, **params):
        self.typ = typ
        if len(typ) > 1:
            raise ValueError(
                "Spider boxes can only have len(typ) == 1, "
                "try Diagram.spiders instead.")
        name = "Spider({}, {}, {})".format(n_legs_in, n_legs_out, typ)
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        cup_like = (n_legs_in, n_legs_out) in ((2, 0), (0, 2))
        params = dict(dict(
            draw_as_spider=not cup_like,
            draw_as_wires=cup_like,
            color="black", drawing_name=""), **params)
        Box.__init__(self, name, dom, cod, **params)

    def __repr__(self):
        return "Spider({}, {}, {})".format(
            len(self.dom), len(self.cod), repr(self.typ))

    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), self.typ)

    def decompose(self):
        return Spider._decompose_spiders(len(self.dom), len(self.cod),
                                         self.typ)

    @staticmethod
    def _decompose_spiders(n_legs_in, n_legs_out, typ):
        if n_legs_out > n_legs_in:
            return Spider._decompose_spiders(n_legs_out, n_legs_in,
                                             typ).dagger()

        if n_legs_in == 0:
            return Id(typ)

        if n_legs_out > 1:
            return (Spider._decompose_spiders(n_legs_in, 1, typ)
                    >> Spider._decompose_spiders(n_legs_out, 1,
                                                 typ).dagger())

        if n_legs_in == 2:
            return Spider(2, n_legs_out, typ)

        if n_legs_in % 2 == 1:
            return (Spider._decompose_spiders(n_legs_in - 1, 1, typ)
                    @ Id(typ) >> Spider(2, n_legs_out, typ))

        new_in = n_legs_in // 2
        return (Spider._decompose_spiders(new_in, 1, typ)
                @ Spider._decompose_spiders(new_in, 1, typ)
                >> Spider(2, n_legs_out, typ))

    @property
    def l(self):
        return type(self)(len(self.dom), len(self.cod), self.typ.l)

    @property
    def r(self):
        return type(self)(len(self.dom), len(self.cod), self.typ.r)


class Functor(monoidal.Functor):
    """
    Implements rigid monoidal functors, i.e. preserving cups and caps.

    >>> s, n = Ty('s'), Ty('n')
    >>> Alice, Bob = Box("Alice", Ty(), n), Box("Bob", Ty(), n)
    >>> loves = Box('loves', Ty(), n.r @ s @ n.l)
    >>> love_box = Box('loves', n @ n, s)
    >>> ob = {s: s, n: n}
    >>> ar = {Alice: Alice, Bob: Bob}
    >>> ar.update({loves: Cap(n.r, n) @ Cap(n, n.l)
    ...                   >> Id(n.r) @ love_box @ Id(n.l)})
    >>> F = Functor(ob, ar)
    >>> sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    >>> assert F(sentence).normal_form() == Alice >> Id(n) @ Bob >> love_box
    >>> from discopy import drawing
    >>> drawing.equation(
    ...     sentence, F(sentence), symbol='$\\\\mapsto$', figsize=(5, 2),
    ...     path='docs/_static/imgs/rigid/functor-example.png')

    .. image:: ../_static/imgs/rigid/functor-example.png
        :align: center
    """
    def __init__(self, ob, ar, ob_factory=Ty, ar_factory=Diagram):
        super().__init__(ob, ar, ob_factory=ob_factory, ar_factory=ar_factory)

    def __call__(self, diagram):
        if isinstance(diagram, monoidal.Ty):
            def adjoint(obj):
                if not hasattr(obj, "z") or not obj.z:
                    return self.ob[type(diagram)(obj)]
                result = self.ob[type(diagram)(type(obj)(obj.name, z=0))]
                if obj.z < 0:
                    for _ in range(-obj.z):
                        result = result.l
                elif obj.z > 0:
                    for _ in range(obj.z):
                        result = result.r
                return result
            return self.ob_factory().tensor(*map(adjoint, diagram.objects))
        if isinstance(diagram, Cup):
            return self.ar_factory.cups(
                self(diagram.dom[:1]), self(diagram.dom[1:]))
        if isinstance(diagram, Cap):
            return self.ar_factory.caps(
                self(diagram.cod[:1]), self(diagram.cod[1:]))
        if isinstance(diagram, Spider):
            return self.ar_factory.spiders(
                len(diagram.dom), len(diagram.cod), self(diagram.typ))
        if isinstance(diagram, Box):
            if not hasattr(diagram, "z") or not diagram.z:
                return super().__call__(diagram)
            z = diagram.z
            for _ in range(abs(z)):
                diagram = diagram.l if z > 0 else diagram.r
            result = super().__call__(diagram)
            for _ in range(abs(z)):
                result = result.l if z < 0 else result.r
            return result
        if isinstance(diagram, monoidal.Diagram):
            return super().__call__(diagram)
        raise TypeError(messages.type_err(Diagram, diagram))


def cups(left, right, ar_factory=Diagram, cup_factory=Cup, reverse=False):
    """ Constructs a diagram of nested cups. """
    for typ in left, right:
        if not isinstance(typ, Ty):
            raise TypeError(messages.type_err(Ty, typ))
    if left.r != right and right.r != left:
        raise AxiomError(messages.are_not_adjoints(left, right))
    result = ar_factory.id(left @ right)
    for i in range(len(left)):
        j = len(left) - i - 1
        cup = cup_factory(left[j:j + 1], right[i:i + 1])
        layer = ar_factory.id(left[:j]) @ cup @ ar_factory.id(right[i + 1:])
        result = result << layer if reverse else result >> layer
    return result


def caps(left, right, ar_factory=Diagram, cap_factory=Cap):
    """ Constructs a diagram of nested caps. """
    return cups(left, right, ar_factory, cap_factory, reverse=True)


def spiders(
        n_legs_in, n_legs_out, typ,
        ar_factory=Diagram, spider_factory=Spider):
    """ Constructs a diagram of interleaving spiders. """
    id, swap, spider = ar_factory.id, ar_factory.swap, spider_factory
    ts = [typ[i:i + 1] for i in range(len(typ))]
    result = id().tensor(*[spider(n_legs_in, n_legs_out, t) for t in ts])

    for i, t in enumerate(ts):
        for j in range(n_legs_in - 1):
            result <<= id(result.dom[:i * j + i + j]) @ swap(
                t, result.dom[i * j + i + j:i * n_legs_in + j]
            ) @ id(result.dom[i * n_legs_in + j + 1:])

        for j in range(n_legs_out - 1):
            result >>= id(result.cod[:i * j + i + j]) @ swap(
                result.cod[i * j + i + j:i * n_legs_out + j], t
            ) @ id(result.cod[i * n_legs_out + j + 1:])
    return result
