# -*- coding: utf-8 -*-

"""
The free rigid category, i.e. diagrams with cups and caps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Ty
    PRO
    Diagram
    Box
    Cup
    Cap
    Sum
    Category
    Functor

Axioms
------

>>> unit, s, n = Ty(), Ty('s'), Ty('n')
>>> t = n.r @ s @ n.l
>>> assert t @ unit == t == unit @ t
>>> assert t.l.r == t == t.r.l
>>> left_snake, right_snake = Id(n.r).transpose(left=True), Id(n.l).transpose()
>>> assert left_snake.normal_form() == Id(n) == right_snake.normal_form()

>>> from discopy.drawing import Equation
>>> Equation(left_snake, Id(n), right_snake).draw(
...     figsize=(4, 2), path='docs/_static/rigid/typed-snake-equation.png')

.. image:: /_static/rigid/typed-snake-equation.png
    :align: center
"""

from __future__ import annotations

from collections.abc import Callable

from typing import Iterator

from discopy import cat, monoidal, closed, messages
from discopy.cat import factory
from discopy.utils import (
    assert_isinstance,
    factory_name,
    BinaryBoxConstructor,
    AxiomError,
    assert_isatomic
)


class Ob(cat.Ob):
    """
    A rigid object has adjoints :meth:`Ob.l` and :meth:`Ob.r`.

    Parameters:
        name : The name of the object.
        z : The winding number.

    Example
    -------
    >>> a = Ob('a')
    >>> assert a.l.r == a.r.l == a and a != a.l.l != a.r.r
    """
    __ambiguous_inheritance__ = True

    def __setstate__(self, state):
        if '_z' in state:  # Backward compatibility
            self.z = state['_z']
            del state['_z']
        super().__setstate__(state)

    def __init__(self, name: str, z: int = 0):
        assert_isinstance(z, int)
        self.z = z
        super().__init__(name)

    @property
    def l(self) -> Ob:
        """ The left adjoint of the object. """
        return type(self)(self.name, self.z - 1)

    @property
    def r(self) -> Ob:
        """ The right adjoint of the object. """
        return type(self)(self.name, self.z + 1)

    def __eq__(self, other):
        return cat.Ob.__eq__(self, other) and self.z == other.z

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return factory_name(type(self))\
            + f"({repr(self.name)}{', z=' + repr(self.z) if self.z else ''})"

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


@factory
class Ty(closed.Ty):
    """
    A rigid type is a closed type with rigid objects inside.

    Parameters:
        inside (tuple[Ob, ...]) : The objects inside the type.

    Example
    -------
    >>> s, n = Ty('s'), Ty('n')
    >>> assert n.l.r == n == n.r.l
    >>> assert (s @ n).l == n.l @ s.l and (s @ n).r == n.r @ s.r
    """
    def __setstate__(self, state):
        if '_z' in state:  # Backward compatibility
            del state['_z']
        super().__setstate__(state)

    def assert_isadjoint(self, other):
        """
        Raise ``AxiomError`` if two rigid types are not adjoints.

        Parameters:
            other : The alleged right adjoint.
        """
        if self.r != other and self != other.r:
            raise AxiomError(messages.NOT_ADJOINT.format(self, other))
        if self.r != other:
            raise AxiomError(messages.NOT_RIGID_ADJOINT.format(self, other))

    @property
    def l(self) -> Ty:
        """ The left adjoint of the type. """
        return self.factory(*[x.l for x in self.inside[::-1]])

    @property
    def r(self) -> Ty:
        """ The right adjoint of the type. """
        return self.factory(*[x.r for x in self.inside[::-1]])

    @property
    def z(self) -> int:
        """ The winding number is only defined for types of length 1. """
        assert_isatomic(self)
        return self.inside[0].z

    def __lshift__(self, other):
        return self @ other.l

    def __rshift__(self, other):
        return self.r @ other

    ob_factory = Ob


@factory
class PRO(monoidal.PRO, Ty):
    """
    A rigid PRO is a natural number ``n`` seen as a rigid type of length ``n``.

    Parameters
    ----------
    n : int
        The length of the PRO type.
    """
    __ambiguous_inheritance__ = (monoidal.PRO, )
    l = r = property(lambda self: self)


class Layer(monoidal.Layer):
    """
    A rigid layer is a monoidal layer that can be rotated.

    Parameters:
        left : The type on the left of the layer.
        box : The box in the middle of the layer.
        right : The type on the right of the layer.
        more : More boxes and types to the right,
               used by :meth:`Diagram.foliation`.
    """
    def rotate(self, left=False):
        return type(self)(*(x.l if left else x.r for x in list(self)[::-1]))

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))


@factory
class Diagram(closed.Diagram):
    """
    A rigid diagram is a closed diagram
    with :class:`Cup` and :class:`Cap` boxes.

    Parameters:
        inside (tuple[Layer, ...]) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.

    Example
    -------
    >>> I, n, s = Ty(), Ty('n'), Ty('s')
    >>> Alice, jokes = Box('Alice', I, n), Box('jokes', I, n.r @ s)
    >>> d = Alice >> Id(n) @ jokes >> Cup(n, n.r) @ Id(s)
    >>> d.draw(figsize=(3, 2),
    ...        path='docs/_static/rigid/diagram-example.png')

    .. image:: /_static/rigid/diagram-example.png
        :align: center
    """
    __ambiguous_inheritance__ = True

    ty_factory = Ty
    layer_factory = Layer

    over = staticmethod(lambda base, exponent: base << exponent)
    under = staticmethod(lambda base, exponent: exponent >> base)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left=True) -> Diagram:
        return base @ cls.cups(exponent.l, exponent) if left\
            else cls.cups(exponent, exponent.r) @ base

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> Diagram:
        """
        Construct a diagram of nested cups for types ``left`` and ``right``.

        Parameters:
            left : The type left of the cup.
            right : Its right adjoint.

        Example
        -------
        >>> a, b = Ty('a'), Ty('b')
        >>> Diagram.cups(a.l @ b, b.r @ a).draw(figsize=(3, 1),\\
        ... margins=(0.3, 0.05), path='docs/_static/rigid/cups.png')

        .. image:: /_static/rigid/cups.png
            :align: center
        """
        return nesting(cls, cls.cup_factory)(left, right)

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> Diagram:
        """
        Construct a diagram of nested caps for types ``left`` and ``right``.

        Parameters:
            left : The type left of the cap.
            right : Its left adjoint.

        Example
        -------
        >>> a, b = Ty('a'), Ty('b')
        >>> Diagram.caps(a.r @ b, b.l @ a).draw(figsize=(3, 1),\\
        ... margins=(0.3, 0.05), path='docs/_static/rigid/caps.png')

        .. image:: /_static/rigid/caps.png
            :align: center
        """
        return nesting(cls, cls.cap_factory)(left, right)

    def curry(self, n=1, left=True) -> Diagram:
        """
        The curry of a rigid diagram is obtained using cups and caps.

        >>> from discopy.drawing import Equation as Eq
        >>> x = Ty('x')
        >>> g = Box('g', x @ x, x)
        >>> Eq(Eq(g.curry(left=False), g, symbol="$\\\\mapsfrom$"),
        ...     g.curry(), symbol="$\\\\mapsto$").draw(
        ...         path="docs/_static/rigid/curry.png")

        .. image:: /_static/rigid/curry.png
            :align: center
        """
        if left:
            base, exponent = self.dom[:-n], self.dom[-n:]
            return base @ self.caps(exponent, exponent.l) >> self @ exponent.l
        base, exponent = self.dom[n:], self.dom[:n]
        return self.caps(exponent.r, exponent) @ base >> exponent.r @ self

    def rotate(self, left=False):
        """
        The half-turn rotation of a diagram, called with ``.l`` and ``.r``.

        Example
        -------
        >>> from discopy import drawing
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', Ty(), x)
        >>> g = Box('g', Ty(), x.r @ y)
        >>> diagram = f @ g >> Cup(x, x.r) @ y
        >>> LHS = drawing.Equation(diagram.l, diagram, symbol="$\\\\mapsfrom$")
        >>> RHS = drawing.Equation(LHS, diagram.r, symbol="$\\\\mapsto$")
        >>> RHS.draw(figsize=(8, 3), path='docs/_static/rigid/rotate.png')

        .. image:: /_static/rigid/rotate.png
            :align: center
        """
        dom, cod = (x.l if left else x.r for x in (self.cod, self.dom))
        inside = tuple(
            layer.l if left else layer.r for layer in self.inside[::-1])
        return self.factory(inside, dom, cod, _scan=False)

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))

    def transpose(self, left=False):
        """
        The transpose of a diagram, i.e. its composition with cups and caps.

        Parameters:
            left : Whether to transpose left or right.

        Example
        -------
        >>> from discopy.drawing import Equation
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
        >>> LHS = Equation(f.transpose(left=True), f, symbol="$\\\\mapsfrom$")
        >>> RHS = Equation(LHS, f.transpose(), symbol="$\\\\mapsto$")
        >>> RHS.draw(figsize=(8, 3), path="docs/_static/rigid/transpose.png")

        .. image:: /_static/rigid/transpose.png
        """
        if left:
            return self.cod.l @ self.caps(self.dom, self.dom.l)\
                >> self.cod.l @ self @ self.dom.l\
                >> self.cups(self.cod.l, self.cod) @ self.dom.l
        return self.caps(self.dom.r, self.dom) @ self.cod.r\
            >> self.dom.r @ self @ self.cod.r\
            >> self.dom.r @ self.cups(self.cod, self.cod.r)

    def transpose_box(self, i, j=0, left=False):
        """
        Transpose the box at index ``i``.

        Parameters:
            i : The vertical index of the box to transpose.
            j : The horizontal index of the box to transpose, only needed if
                the layer ``i`` has more than one box.
            left : Whether to transpose left or right.

        Example
        -------
        >>> from discopy.drawing import Equation
        >>> x, y, z = Ty(*"xyz")
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> d = (f @ g).foliation()
        >>> transpose_l = d.transpose_box(0, 0, left=True)
        >>> transpose_r = d.transpose_box(0, 1, left=False)
        >>> LHS = Equation(transpose_l, d, symbol="$\\\\mapsfrom$")
        >>> RHS = Equation(LHS, transpose_r, symbol="$\\\\mapsto$")
        >>> RHS.draw(
        ...     figsize=(8, 3), path="docs/_static/rigid/transpose_box.png")

        .. image:: /_static/rigid/transpose_box.png
        """
        box = list(self.inside[i])[2 * j + 1]
        transposed_box = (box.r if left else box.l).transpose(left)
        top, bottom = self[:i], self[i + 1:]
        left_boxes_and_types = list(self.inside[i])[:2 * j + 1]
        right_boxes_and_types = list(self.inside[i])[2 * j + 2:]
        left_layer, right_layer = [
            self.id().tensor(
                *(x if k % 2 else self.id(x) for k, x in enumerate(xs)))
            for xs in [left_boxes_and_types, right_boxes_and_types]]
        return top >> left_layer @ transposed_box @ right_layer >> bottom

    def snake_removal(self, left=False) -> Iterator[Diagram]:
        """
        Return a generator which yields normalization steps.

        Parameters:
            left : Passed to :meth:`discopy.monoidal.Diagram.normalize`.

        Example
        -------
        >>> from discopy.rigid import *
        >>> n, s = Ty('n'), Ty('s')
        >>> cup, cap = Cup(n, n.r), Cap(n.r, n)
        >>> f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
        >>> diagram = g @ cap >> f[::-1] @ Id(n.r) @ f >> cup @ h
        >>> for d in diagram.normalize(): print(d)
        g >> f[::-1] >> n @ Cap(n.r, n) >> n @ n.r @ f >> Cup(n, n.r) @ n >> h
        g >> f[::-1] >> n @ Cap(n.r, n) >> Cup(n, n.r) @ n >> f >> h
        g >> f[::-1] >> f >> h
        """
        from discopy import monoidal
        from discopy.rigid import Cup, Cap

        def follow_wire(diagram, i, j):
            """
            Given a diagram, the index of a box i and the offset j of an output
            wire, returns (i, j, obstructions) where:
            - i is the index of the box which takes this wire as input, or
            len(diagram) if it is connected to the bottom boundary.
            - j is the offset of the wire at its bottom end.
            - obstructions is a pair of lists of indices for the boxes on
            the left and right of the wire we followed.
            """
            left_obstruction, right_obstruction = [], []
            while i < len(diagram) - 1:
                i += 1
                box, off = diagram.boxes[i], diagram.offsets[i]
                if off <= j < off + len(box.dom):
                    return i, j, (left_obstruction, right_obstruction)
                if off <= j:
                    j += len(box.cod) - len(box.dom)
                    left_obstruction.append(i)
                else:
                    right_obstruction.append(i)
            return len(diagram), j, (left_obstruction, right_obstruction)

        def find_snake(diagram):
            """
            Given a diagram, returns (cup, cap, obstructions, left_snake)
            if there is a yankable pair, otherwise returns None.
            """
            for cap in range(len(diagram)):
                if not isinstance(diagram.boxes[cap], Cap):
                    continue
                for left_snake, wire in [(True, diagram.offsets[cap]),
                                         (False, diagram.offsets[cap] + 1)]:
                    cup, wire, obstructions =\
                        follow_wire(diagram, cap, wire)
                    not_yankable =\
                        cup == len(diagram)\
                        or not isinstance(diagram.boxes[cup], Cup)\
                        or left_snake and diagram.offsets[cup] + 1 != wire\
                        or not left_snake and diagram.offsets[cup] != wire
                    if not_yankable:
                        continue
                    return cup, cap, obstructions, left_snake
            return None

        def unsnake(diagram, cup, cap, obstructions, left_snake=False):
            """
            Given a diagram and the indices for a cup and cap pair
            and a pair of lists of obstructions on the left and right,
            returns a new diagram with the snake removed.

            A left snake is one of the form Id @ Cap >> Cup @ Id.
            A right snake is one of the form Cap @ Id >> Id @ Cup.
            """
            left_obstruction, right_obstruction = obstructions
            if left_snake:
                for box in left_obstruction:
                    diagram = diagram.interchange(box, cap)
                    yield diagram
                    for i, right_box in enumerate(right_obstruction):
                        if right_box < box:
                            right_obstruction[i] += 1
                    cap += 1
                for box in right_obstruction[::-1]:
                    diagram = diagram.interchange(box, cup)
                    yield diagram
                    cup -= 1
            else:
                for box in left_obstruction[::-1]:
                    diagram = diagram.interchange(box, cup)
                    yield diagram
                    for i, right_box in enumerate(right_obstruction):
                        if right_box > box:
                            right_obstruction[i] -= 1
                    cup -= 1
                for box in right_obstruction:
                    diagram = diagram.interchange(box, cap)
                    yield diagram
                    cap += 1
            inside = diagram.inside[:cap] + diagram.inside[cup + 1:]
            yield diagram.factory(
                inside, diagram.dom, diagram.cod, _scan=False)

        diagram = self
        while True:
            yankable = find_snake(diagram)
            if yankable is None:
                break
            for _diagram in unsnake(diagram, *yankable):
                yield _diagram
                diagram = _diagram
        for _diagram in monoidal.Diagram.normalize(diagram, left=left):
            yield _diagram

    normalize = snake_removal

    def normal_form(self, **params):
        """
        Implements the normalisation of rigid categories,
        see Dunn and Vicary :cite:`DunnVicary19`, definition 2.12.

        Examples
        --------
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
        """
        return super().normal_form(**params)


class Box(closed.Box, Diagram):
    """
    A rigid box is a closed box in a rigid diagram.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. its input.
        cod : The codomain of the box, i.e. its output.
        z : The winding number of the box,
            i.e. the number of half-turn rotations.

    Example
    -------
    >>> a, b = Ty('a'), Ty('b')
    >>> f = Box('f', a, b.l @ b)
    >>> assert f.l.z == -1 and f.z == 0 and f.r.z == 1
    >>> assert f.r.l == f == f.l.r
    >>> assert f.l.l != f != f.r.r
    """
    __ambiguous_inheritance__ = (closed.Box, )

    def __setstate__(self, state):
        if '_z' in state:  # Backward compatibility
            self.z = state['_z']
            del state['_z']
        super().__setstate__(state)

    def __init__(self, name: str, dom: Ty, cod: Ty, data=None, z=0, **params):
        self.z = z
        closed.Box.__init__(self, name, dom, cod, data=data, **params)

    def __str__(self):
        return cat.Box.__str__(self) if not self.z\
            else str(self.r) + '.l' if self.z < 0 else str(self.l) + '.r'

    def __repr__(self):
        if self.is_dagger:
            return closed.Box.__repr__(self)
        return closed.Box.__repr__(self)[:-1] + (
            f', z={self.z})' if self.z else ')')

    def __eq__(self, other):
        if isinstance(other, Box):
            return cat.Box.__eq__(self, other) and self.z == other.z
        return monoidal.Box.__eq__(self, other)

    def __hash__(self):
        return hash(repr(self))

    def rotate(self, left=False):
        dom, cod = (
            getattr(x, 'l' if left else 'r') for x in (self.cod, self.dom))
        z = self.z + (-1 if left else 1)
        return type(self)(self.name, dom=dom, cod=cod,
                          data=self.data, is_dagger=self.is_dagger, z=z)

    @property
    def is_transpose(self):
        """ Whether the box is an odd rotation of a generator. """
        return not self.is_dagger and self.z and bool(self.z % 2)

    def to_drawing(self):
        result = super().to_drawing()
        result.is_transpose = self.is_transpose
        return result


class Sum(closed.Sum, Box):
    """
    A rigid sum is a closed sum that can be transposed.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """
    __ambiguous_inheritance__ = (closed.Sum, )

    def rotate(self, left=False) -> Sum:
        if left:
            return self.sum_factory(
                tuple(term.l for term in self.terms), self.cod.l, self.dom.l)
        return self.sum_factory(
            tuple(term.r for term in self.terms), self.cod.r, self.dom.r)


class Cup(BinaryBoxConstructor, Box):
    """
    The counit of the adjunction for an atomic type.

    Parameters:
        left : The atomic type.
        right : Its right adjoint.

    Example
    -------
    >>> n = Ty('n')
    >>> Cup(n, n.r).draw(figsize=(2,1), margins=(0.5, 0.05),\\
    ... path='docs/_static/rigid/cup.png')

    .. image:: /_static/rigid/cup.png
        :align: center
    """
    def __init__(self, left: Ty, right: Ty):
        assert_isatomic(left, Ty)
        assert_isatomic(right, Ty)
        left.assert_isadjoint(right)
        name = f"Cup({left}, {right})"
        dom, cod = left @ right, self.ty_factory()
        BinaryBoxConstructor.__init__(self, left, right)
        Box.__init__(self, name, dom, cod, draw_as_wires=True)

    def rotate(self, left=False):
        return self.cap_factory(self.right.l, self.left.l) if left\
            else self.cap_factory(self.right.r, self.left.r)

    def dagger(self):
        """
        The dagger of a rigid cup is ill-defined,
        use a :class:`pivotal.Cup` instead.
        """
        raise AxiomError("Rigid cups have no dagger, use pivotal instead.")


class Cap(BinaryBoxConstructor, Box):
    """
    The unit of the adjunction for an atomic type.

    Parameters:
        left : The atomic type.
        right : Its left adjoint.

    Example
    -------
    >>> n = Ty('n')
    >>> Cap(n, n.l).draw(figsize=(2,1), margins=(0.5, 0.05),\\
    ... path='docs/_static/rigid/cap.png')

    .. image:: /_static/rigid/cap.png
        :align: center
    """
    def __init__(self, left: Ty, right: Ty):
        assert_isatomic(left, Ty)
        assert_isatomic(right, Ty)
        right.assert_isadjoint(left)
        name = f"Cap({left}, {right})"
        dom, cod = self.ty_factory(), left @ right
        BinaryBoxConstructor.__init__(self, left, right)
        Box.__init__(self, name, dom, cod, draw_as_wires=True)

    def rotate(self, left=False):
        return self.cup_factory(self.right.l, self.left.l) if left\
            else self.cup_factory(self.right.r, self.left.r)

    def dagger(self):
        """
        The dagger of a rigid cap is ill-defined,
        use a :class:`pivotal.Cap` instead.
        """
        raise AxiomError("Rigid caps have no dagger, use pivotal instead.")


class Category(closed.Category):
    """
    A rigid category is a monoidal category
    with methods :code:`l`, :code:`r`, :code:`cups` and :code:`caps`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = Ty, Diagram


class Functor(closed.Functor):
    """
    A rigid functor is a closed functor that preserves cups and caps.

    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.

    Example
    -------
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice, Bob = Box("Alice", Ty(), n), Box("Bob", Ty(), n)
    >>> loves = Box('loves', Ty(), n.r @ s @ n.l)
    >>> love_box = Box('loves', n @ n, s)
    >>> ob = {s: s, n: n}
    >>> ar = {Alice: Alice, Bob: Bob}
    >>> ar.update({loves: Cap(n.r, n) @ Cap(n, n.l) >> n.r @ love_box @ n.l})
    >>> F = Functor(ob, ar)
    >>> sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ s @ Cup(n.l, n)
    >>> assert F(sentence).normal_form() == Alice >> Id(n) @ Bob >> love_box

    >>> from discopy.drawing import Equation
    >>> Equation(sentence, F(sentence), symbol='$\\\\mapsto$').draw(
    ...     figsize=(5, 2), path='docs/_static/rigid/functor-example.png')

    .. image:: /_static/rigid/functor-example.png
        :align: center
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Ty) or isinstance(other, Ob) and other.z == 0:
            return super().__call__(other)
        if isinstance(other, Ob):
            return self(other.r).l if other.z < 0 else self(other.l).r
        if isinstance(other, Cup):
            return self.cod.ar.cups(self(other.dom[:1]), self(other.dom[1:]))
        if isinstance(other, Cap):
            return self.cod.ar.caps(self(other.cod[:1]), self(other.cod[1:]))
        if isinstance(other, Box):
            if not hasattr(other, "z") or not other.z:
                return super().__call__(other)
            z = other.z
            for _ in range(abs(z)):
                other = other.l if z > 0 else other.r
            result = super().__call__(other)
            for _ in range(abs(z)):
                result = result.l if z < 0 else result.r
            return result
        return super().__call__(other)


def nesting(cls: type, factory: Callable) -> Callable[[Ty, Ty], Diagram]:
    """
    Take a :code:`factory` for cups or caps of atomic types
    and extends it recursively.

    Parameters:
        cls : A diagram factory, e.g. :class:`Diagram`.
        factory :
            A factory for cups (or caps) of atomic types, e.g. :class:`Cup`.
    """
    def method(left: Ty, right: Ty) -> Diagram:
        if len(left) == 0:
            return cls.id(left[:0])
        head, tail = factory(left[0], right[-1]), method(left[1:], right[:-1])
        if head.dom:  # We are nesting cups.
            return left[0] @ tail @ right[-1] >> head
        return head >> left[0] @ tail @ right[-1]

    return method


Diagram.cup_factory, Diagram.cap_factory, Diagram.sum_factory = Cup, Cap, Sum

Id = Diagram.id
