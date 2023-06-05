# -*- coding: utf-8 -*-

"""
The free compact category on a symmetric traced category, or more generally the
free ribbon category on a balanced traced category.

Concretely, this is a "glorification of the construction of the integers from
the natural numbers". This so-called Int-construction first appeared in Joyal,
Street & Verity :cite:p:`JoyalEtAl96`. It is sometimes called the "geometry of
interaction" construction, see :cite:t:`Abramsky96`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Diagram
    NamedGeneric

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        Int

Example
-------

>>> from discopy.grammar import pregroup
>>> from discopy.grammar.pregroup import Word, Cup, Diagram, Functor
>>> s, n = map(pregroup.Ty, "sn")
>>> Alice, loves, Bob\\
...     = Word('Alice', n), Word('loves', n.r @ s @ n.l), Word('Bob', n)
>>> who = Word('who', n.r @ n @ (n.r @ s).l)
>>> noun_phrase = Alice @ who @ loves @ Bob\\
...     >> Cup(n, n.r) @ n @ Diagram.cups((n.r @ s).l, n.r @ s) @ Cup(n.l, n)

>>> from discopy.frobenius import Ty as T, Diagram as D, Box, Category, Swap
>>> S, N = map(T, "SN")
>>> F = Functor(
...     ob={s: Ty[T](S), n: Ty[T](N)},
...     ar={Alice: Box('A', T(), N),
...         who: Box('W', S @ N, N @ N),
...         loves: Box('L', N @ N, S),
...         Bob: Box('B', T(), N)},
...     cod=Int(Category(T, D)))
>>> image = F(noun_phrase).inside.to_hypergraph().interchange(1, 3)\\
...     .to_diagram().interchange(1, 2).naturality(2, left=False)

>>> from discopy.drawing import Equation
>>> Equation(noun_phrase, image, symbol="$\\\\mapsto$").draw(
...     figsize=(10, 4), path="docs/_static/int/alice-loves-bob.png")

.. image:: /_static/int/alice-loves-bob.png
    :align: center
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import wraps

from discopy import (
    balanced,
    traced,
    rigid,
    pivotal,
    ribbon,
    messages
)
from discopy.cat import Composable, assert_iscomposable
from discopy.monoidal import Whiskerable
from discopy.utils import (
    NamedGeneric, unbiased, assert_isinstance, factory_name)


@dataclass
class Ty(NamedGeneric['natural']):
    """
    An integer type is a pair of :attr:`natural` types.

    Parameters:
        positive : The positive half of the type.
        negative : The negative half of the type.

    Note
    ----
    Integer types are parameterised by natural types, e.g.

    >>> assert Ty == Ty[pivotal.Ty] and Ty[int].natural == int

    The prefix operator ``-`` reverses positive and negative, e.g.

    >>> x, y, z = map(Ty[int], [1, 2, 3])
    >>> assert x @ -y @ z == Ty[int](1 + 3, 2)
    """
    natural = pivotal.Ty

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

    def __repr__(self):
        pos, neg = repr(self.positive), repr(self.negative)
        return f"interaction.Ty[{factory_name(self.natural)}]"\
               f"(positive={pos}, negative={neg})"

    def __str__(self):
        try:
            return " @ ".join(list(map(str, self.positive)) + [
                f"-{x}" for x in reversed(self.negative)])
        except TypeError:  # e.g. when Ty.natural == int
            return repr(self)

    def tensor(self, *others: Ty):
        if any(not isinstance(other, Ty) for other in others):
            return NotImplemented
        unit = type(self).natural()
        positive = sum([x.positive for x in (self, ) + others], unit)
        negative = sum([x.negative for x in reversed((self, ) + others)], unit)
        return type(self)(positive, negative)

    __matmul__ = __add__ = tensor

    def __neg__(self):
        positive, negative = self
        return type(self)(negative, positive)

    l = r = property(__neg__)


@dataclass
class Diagram(Composable[Ty], Whiskerable, NamedGeneric['natural']):
    """
    An integer diagram from ``x`` to ``y`` is a :attr:`natural` diagram
    from ``x.positive @ y.negative`` to ``x.negative @ y.positive``.

    Parameters:
        inside : The natural diagram inside.
        dom : The domain of the diagram, i.e. its input.
        cod : The codomain of the diagram, i.e. its output.

    Note
    ----
    By default we take ``natural = ribbon.Diagram`` but this can be any class
    with the methods for a balanced traced category. For example, the category
    of boolean matrices with the direct sum has a trace given by reflexive
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
            raise ValueError(messages.WRONG_DOM.format(
                dom.positive + cod.negative, inside.dom))
        if inside.cod != cod.positive + dom.negative:
            raise ValueError(messages.WRONG_COD.format(
                cod.positive + dom.negative, inside.cod))
        self.inside, self.dom, self.cod = inside, dom, cod

    @unbiased
    def then(self, other: Diagram):
        """
        The composition of two integer diagrams.

        Parameters:
            other : The other diagram with which to compose.

        Example
        -------

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
            >> z @ braid(v, u)).trace(n=v if isinstance(v, int) else len(v))
        return type(self)(inside, dom, cod)

    @classmethod
    def id(cls, dom: Ty = None) -> Diagram:
        """
        The identity on an integer type.

        Parameters:
            dom : The integer type on which to take the identity.

        Example
        -------

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
        dom = Ty[cls.natural.ty_factory]() if dom is None else dom
        positive, negative = dom
        inside = cls.natural.id(positive) @ cls.natural.twist(negative)
        return cls(inside, dom, dom)

    @unbiased
    def tensor(self, other):
        """
        The tensor of two integer diagrams.

        Parameters:
            other : The other diagram to tensor.

        Example
        -------

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
        _braid = self.natural.braid
        inside = _braid(x, x_) @ _braid(v, v_).dagger()\
            >> x_ @ self.inside @ v_\
            >> _braid(y, x_).dagger() @ _braid(v_, u).dagger()\
            >> y @ other.inside @ u
        return type(self)(inside, self.dom @ other.dom, self.cod @ other.cod)

    @classmethod
    def braid(cls, left: Ty, right: Ty) -> Diagram:
        """
        The braid of integer diagrams is given by the following diagram:

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, u, y, v = map(Ty[T], "xuyv")
        >>> Diagram.braid(x @ -u, y @ -v).draw(
        ...     path="docs/_static/int/braid.png")

        .. image:: /_static/int/braid.png
            :align: center

        Parameters:
            left : The left input of the braid.
            right : The right input of the braid.
        """
        _braid, _twist = cls.natural.braid, cls.natural.twist
        x, u = left
        y, v = right
        braids = _braid(x, y) @ _braid(v, u).dagger()\
            >> y @ (_braid(v, x).dagger() >> _braid(x, v).dagger()) @ u
        twists = y @ x @ _twist(v).dagger() @ _twist(u).dagger()
        return cls(braids >> twists, left @ right, right @ left)

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> Diagram:
        """
        The integer cups are given by natural identities.

        Parameters:
            left : The left-hand side of the cups.
            right : The right-hand side of the cups.

        Example
        -------

        This is what the snake equations look like:

        >>> from discopy.drawing import Equation
        >>> x = Ty('x')
        >>> Equation(
        ...     Diagram.caps(x, -x) @ x >> x @ Diagram.cups(-x, x),
        ...     Diagram.id(x),
        ...     x @ Diagram.caps(-x, x) >> Diagram.cups(x, -x) @ x).draw(
        ...         path="docs/_static/int/int-snake-equations.png")

        .. image:: /_static/int/int-snake-equations.png
            :align: center
        """
        rigid.Ty.assert_isadjoint(left, right)
        inside = cls.natural.id(left.positive + left.negative)
        return cls(inside, left @ right, type(left)())

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> Diagram:
        """
        The integer caps are given by natural identities.

        Parameters:
            left : The left-hand side of the caps.
            right : The right-hand side of the caps.
        """
        rigid.Ty.assert_isadjoint(right, left)
        inside = cls.natural.id(left.negative + left.positive)
        return cls(inside, type(left)(), left @ right)

    def dagger(self):
        """
        The dagger of an integer diagram is given by the dagger of its inside.

        >>> from discopy.ribbon import Ty as T, Diagram as D, Box as B
        >>> x, y, u, v = map(Ty[T], "xyuv")
        >>> f = Diagram[D](B('f', T('x', 'v'), T('y', 'u')), x @ -u, y @ -v)
        >>> from discopy.drawing import Equation
        >>> Equation(f, f[::-1], symbol="$\\\\mapsto$").draw(
        ...     path="docs/_static/int/dagger.png")

        .. image:: /_static/int/dagger.png
            :align: center
        """
        return type(self)(self.inside.dagger(), self.cod, self.dom)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        raise IndexError

    def draw(self, **params):
        """ The drawing of an integer diagram is the drawing of its inside. """
        return self.inside.draw(**params)

    def simplify(self):
        """
        Simplify by going back and forth to :class:`Hypergraph`.

        Example
        -------
        >>> from discopy import frobenius
        >>> x = Ty[frobenius.Ty]('x')
        >>> D = Diagram[frobenius.Diagram]
        >>> left_snake = D.id(-x).transpose(left=True)
        >>> right_snake = D.id(-x).transpose(left=False)
        >>> assert left_snake.simplify() == D.id(x) == right_snake.simplify()

        >>> from discopy.drawing import Equation
        >>> Equation(left_snake, Equation(
        ...     D.id(x), right_snake, symbol="$\\\\leftarrow$"),
        ...         symbol="$\\\\rightarrow$").draw(
        ...             path="docs/_static/int/simplify.png")

        .. image:: /_static/int/simplify.png
            :align: center
        """
        return type(self)(self.inside.simplify(), self.dom, self.cod)

    @wraps(balanced.Diagram.naturality)
    def naturality(self, i: int, left=True, down=True, braid=None) -> Diagram:
        return type(self)(
            self.inside.naturality(i, left, down, braid), self.dom, self.cod)

    trace = traced.Diagram.trace
    trace_factory = classmethod(pivotal.Diagram.trace_factory.__func__)
    transpose = rigid.Diagram.transpose
    boxes = property(lambda self: self.inside.boxes)
    to_drawing = lambda self: self.inside.to_drawing()


def Int(category: traced.Category) -> ribbon.Category:
    """
    The Int construction, returns a ribbon category.

    Parameters:
        category : A balanced traced category.

    Example
    -------
    >>> from discopy.ribbon import Ty as T, Diagram as D, Category
    >>> assert Int(Category(T, D)) == Category(Ty[T], Diagram[D])
    """
    return ribbon.Category(Ty[category.ob], Diagram[category.ar])


Id = Diagram.id
