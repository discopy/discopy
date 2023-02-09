# -*- coding: utf-8 -*-

"""
The Int construction of Joyal, Street & Verity :cite:t:`JoyalEtAl96`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Diagram
"""

from __future__ import annotations
from dataclasses import dataclass

from discopy import traced, rigid, pivotal, ribbon
from discopy.cat import Composable, assert_iscomposable
from discopy.monoidal import Whiskerable
from discopy.rigid import assert_isadjoint
from discopy.utils import NamedGeneric, mmap, assert_isinstance


@dataclass
class Ty(NamedGeneric('natural')):
    """
    An integer type is a pair of ``natural`` types,
    by default :class:`ribbon.Ty`.

    Parameters:
        positive : The positive half of the type.
        negative : The negative half of the type.

    Note
    ----
    The prefix operator ``-`` reverses positive and negative, e.g.

    >>> x, y, z = map(Ty[int], [1, 2, 3])
    >>> assert x @ -y @ z == Ty[int](1 + 3, 2)
    """
    natural = ribbon.Ty

    positive: natural
    negative: natural

    def __init__(self, positive: natural = None, negative: natural = None):
        positive, negative = (
            self.natural() if x is None else x for x in (positive, negative))
        positive, negative = (
            x if isinstance(x, type(self).natural) else type(self).natural(x)
            for x in (positive, negative))
        self.positive, self.negative = positive, negative

    def __iter__(self):
        yield self.positive
        yield self.negative

    def tensor(self, *others: Ty):
        if any(not isinstance(other, Ty) for other in others):
            return NotImplemented
        unit = type(self).natural()
        positive = sum([x.positive for x in (self, ) + others], unit)
        negative = sum([x.negative for x in reversed((self, ) + others)], unit)
        return type(self)(positive, negative)

    __matmul__ = tensor

    def __neg__(self):
        positive, negative = self
        return type(self)(negative, positive)

    l = r = property(__neg__)


@dataclass
class Diagram(Composable[Ty], Whiskerable, NamedGeneric('natural')):
    """
    An integer diagram from ``x`` to ``y`` is a ``natural`` diagram
    from ``x.positive @ y.negative`` to ``x.negative @ y.positive``.

    Parameters:
        inside : The natural diagram inside.
        dom : The domain of the diagram, i.e. its input.
        cod : The codomain of the diagram, i.e. its output.

    Note
    ----
    We take ``natural = ribbon.Diagram`` but this can be any class with the
    methods for a balanced traced category. For example, the category of
    boolean matrices with the direct sum has a trace given by reflexive
    transitive closure. We can use it to check the snake equations:

    >>> from discopy.matrix import Matrix
    >>> T, D = Ty[int], Diagram[Matrix[bool]]
    >>> assert D.id(T(2, 2)).transpose()\\
    ...     == D.id(T(2, 2))\\
    ...     == D.id(T(2, 2)).transpose(left=True)
    """
    natural = ribbon.Diagram

    inside: natural
    dom: Ty
    cod: Ty

    def __init__(self, inside: natural, dom: Ty, cod: Ty):
        assert_isinstance(inside, self.natural)
        if inside.dom != dom.positive + cod.negative:
            raise ValueError
        if inside.cod != cod.positive + dom.negative:
            raise ValueError
        self.inside, self.dom, self.cod = inside, dom, cod

    @mmap
    def then(self, other: Diagram):
        """
        The composition of two integer diagrams is given by:

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> u, v, w, x, y, z = map(Ty[T], "uvwxyz")
        >>> f = Diagram[D](B('f', T('x', 'v'), T('y', 'u')), x @ -u, y @ -v)
        >>> g = Diagram[D](B('g', T('y', 'w'), T('z', 'v')), y @ -v, z @ -w)
        >>> (f >> g).draw(path='docs/_static/int/composition.png')

        .. image:: /_static/int/composition.png
            :align: center
        """
        assert_iscomposable(self, other)
        x, u = self.dom
        y, v = self.cod
        z, w = other.cod
        braid = self.natural.braid
        dom, cod = type(self.dom)(x, u), type(self.cod)(z, w)
        inside = (
            x @ braid(w, v)
            >> self.inside @ w
            >> y @ braid(w, u).dagger()
            >> other.inside @ u
            >> z @ braid(v, u)).trace(v if isinstance(v, int) else len(v))
        return type(self)(inside, dom, cod)

    @classmethod
    def id(cls, dom: Ty) -> Diagram:
        """
        The identity on an integer type is the identity on its positive half
        tensored with a twist on its negative half. For example:

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, y, u, v = map(Ty[T], "xyuv")
        >>> f = Diagram[D](B('f', T('x', 'v'), T('y', 'u')), x @ -u, y @ -v)
        >>> (Diagram[D].id(x @ -u) >> f).draw(path='docs/_static/int/idl.png')

        .. image:: /_static/int/idl.png
            :align: center

        >>> (f >> Diagram[D].id(y @ -v)).draw(path='docs/_static/int/idr.png')

        .. image:: /_static/int/idr.png
            :align: center
        """
        positive, negative = dom
        inside = cls.natural.id(positive) @ cls.natural.twist(negative)
        return cls(inside, dom, dom)

    @mmap
    def tensor(self, other):
        """
        The tensor of two integer diagrams is given by:

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, y, u, v = map(Ty[T], "xyuv")
        >>> x_, y_, u_, v_ = map(lambda x: Ty[T](x + '_'), "xyuv")
        >>> f = Diagram[D](B('f', T('x', 'v'), T('y', 'u')), x @ -u, y @ -v)
        >>> f_ = Diagram[D](
        ...     B('f_', T('x_', 'v_'), T('y_', 'u_')), x_ @ -u_, y_ @ -v_)
        >>> (f @ f_).draw(path='docs/_static/int/tensor.png')

        .. image:: /_static/int/tensor.png
            :align: center
        """
        x, u, y, v = tuple(self.dom) + tuple(self.cod)
        x_, u_, y_, v_ = tuple(other.dom) + tuple(other.cod)
        dom = type(self.dom)(x + x_, u_ + u)
        cod = type(self.cod)(y + y_, v_ + v)
        braid = self.natural.braid
        inside = braid(x, x_) @ braid(v, v_).dagger()\
            >> x_ @ self.inside @ v_\
            >> braid(y, x_).dagger() @ braid(v_, u).dagger()\
            >> y @ other.inside @ u
        return type(self)(inside, dom, cod)

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> Diagram:
        """
        The integer cups are given by natural identities.

        This is what the snake equations look like:

        >>> from discopy.drawing import Equation
        >>> Equation(
        ...     Diagram.id(Ty('x')).transpose(),
        ...     Diagram.id(Ty('x')),
        ...     Diagram.id(Ty('x')).transpose(left=True)).draw(
        ...         path="docs/_static/int/int-snake-equations.png")

        .. image:: /_static/int/int-snake-equations.png
            :align: center
        """
        assert_isadjoint(left, right)
        inside = cls.natural.id(left.positive + left.negative)
        return cls(inside, left @ right, type(left)())

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> Diagram:
        """
        The integer caps are given by natural identities.
        """
        assert_isadjoint(left, right)
        inside = cls.natural.id(left.negative + left.positive)
        return cls(inside, type(left)(), left @ right)

    def dagger(self):
        return type(self)(self.inside.dagger(), self.cod, self.dom)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        raise IndexError

    def draw(self, **params):
        """ The drawing of an integer diagram is the drawing of its inside. """
        return self.inside.draw(**params)

    trace = traced.Diagram.trace
    trace_factory = classmethod(pivotal.Diagram.trace_factory.__func__)
    transpose = rigid.Diagram.transpose
    boxes = property(lambda self: self.inside.boxes)
    to_drawing = lambda self: self.inside.to_drawing()
