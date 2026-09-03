# -*- coding: utf-8 -*-

"""
The category of optics over a symmetric underlying `category`, and its
cartesian instance: lenses over a Markov category.

An optic from a pair `(x, x_)` to a pair `(y, y_)` is a residual `m` with a
`forward` morphism `x -> m @ y` and a `backward` morphism `m @ y_ -> x_` in
the underlying category, see :cite:t:`Riley18`. Composition tensors the
residuals, as in the category of parametric maps :mod:`discopy.para`: the
forward leg is a coparametric map with the residual as coparameter, the
backward leg a parametric map with the residual as parameter
:cite:p:`CapucciEtAl21`. When the underlying category is Markov, an optic is a
lens `(get, put)` with the residual normalised to `x` by copying, the
bidirectional accessors of :cite:t:`ClarkeEtAl20`; when it is traced, an
optic is an integer diagram :mod:`discopy.interaction` with the residual as
the wire between the two legs.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Optic
    Lens

Axioms
------

Composition tensors the residuals and routes nothing:

>>> from discopy.symmetric import Ty as T, Box
>>> x, x_, y, y_, z, z_ = map(T, ["x", "x'", "y", "y'", "z", "z'"])
>>> m, n = map(T, "mn")
>>> X, Y, Z = Ty(x, x_), Ty(y, y_), Ty(z, z_)
>>> f = Optic(X, Y, Box('f', x, m @ y), Box("f'", m @ y_, x_), m)
>>> g = Optic(Y, Z, Box('g', y, n @ z), Box("g'", n @ z_, y_), n)
>>> assert (f >> g).residual == m @ n
>>> assert (f >> g).forward == f.forward >> m @ g.forward
>>> assert (f >> g).backward == m @ g.backward >> f.backward
>>> (f >> g).to_int().draw(doctest="docs/_static/optics/then.svg")

.. image:: /_static/optics/then.svg
    :align: center

The tensor swaps the residual of the right-hand side past the left-hand
side, on both legs:

>>> w, w_, k = map(T, ["w", "w'", "k"])
>>> W = Ty(w, w_)
>>> h = Optic(Z, W, Box('h', z, k @ w), Box("h'", k @ w_, z_), k)
>>> assert (f @ h).residual == m @ k
>>> assert (f @ h).dom == X @ Z and (f @ h).cod == Y @ W
>>> (f @ h).to_int().draw(doctest="docs/_static/optics/tensor.svg")

.. image:: /_static/optics/tensor.svg
    :align: center

The identity and swap of :class:`Optic` are those of the underlying
category on each half, with the empty residual:

>>> assert Optic.id(X) == Optic.lift(f.category.id(x), f.category.id(x_))
>>> assert Optic.swap(X, Y).forward == f.category.swap(x, y)
>>> assert Optic.swap(X, Y).backward == f.category.swap(y_, x_)

Two optics are equal when their representatives are; the quotient by
sliding a morphism across the residual is decided by :meth:`Optic.to_int`
in a traced category and by :meth:`Optic.to_lens` in a Markov one.

Example
-------

Lenses over :class:`Function <discopy.python.Function>` are the accessors of
functional programming: `get` reads a part of a structure and `put` writes
it back. The lens on the first component of a pair:

>>> from discopy.python import Function
>>> P, A = Ty[tuple]((int, str), (int, str)), Ty[tuple]((int, ), (int, ))
>>> first = Lens[Function](P, A,
...     Function(lambda a, b: a, (int, str), (int, )),
...     Function(lambda a, b, a_: (a_, b), (int, str, int), (int, str)))
>>> first.get(1, "b")
1
>>> first.put(1, "b", 2)
(2, 'b')

It is well-behaved, i.e. it satisfies the three lens laws:

>>> assert first.get(*first.put(1, "b", 2)) == 2
>>> assert first.put(1, "b", first.get(1, "b")) == (1, "b")
>>> assert first.put(*first.put(1, "b", 2), 3) == first.put(1, "b", 3)

Lenses compose by the chain rule, which is why they are the semantics of
reverse-mode differentiation :cite:p:`CruttwellEtAl22`: the reverse
derivative of a function `f` is the lens with `get` its value and `put`
its Jacobian transposed, applied to the incoming gradient.

>>> R = Ty[tuple]((float, ), (float, ))
>>> square = Lens[Function](R, R,
...     Function(lambda x: x * x, (float, ), (float, )),
...     Function(lambda x, dy: 2 * x * dy, (float, float), (float, )))
>>> (square >> square).get(3.)
81.0
>>> (square >> square).put(3., 1.)
108.0

A neural network is then a parametric lens: the parameters are the weights
and their gradients come back beside the input's, see :mod:`discopy.para`.

>>> from discopy.para import Symmetric
>>> layer = Symmetric[Lens[Function]](R, R, Lens[Function](R @ R, R,
...     Function(lambda x, w: w * x, (float, float), (float, )),
...     Function(lambda x, w, dy: (w * dy, x * dy),
...              (float, float, float), (float, float))), param=R)
>>> network = layer >> layer
>>> assert network.param == R @ R
>>> network.inside.get(2., 3., 5.)
30.0
>>> network.inside.put(2., 3., 5., 1.)
(15.0, 10.0, 6.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_origin

from discopy import interaction, markov, monoidal, symmetric
from discopy.abc import NamedGeneric, SymmetricCategory
from discopy.utils import (
    assert_iscomposable, assert_isinstance, classproperty, factory_name,
    unbiased)


class Ty(interaction.Ty):
    """
    A pair of types of the underlying category, tensored componentwise.

    Parameters:
        positive : The forward half of the type.
        negative : The backward half of the type.

    Note
    ----
    An :class:`interaction.Ty <discopy.interaction.Ty>` reverses the negative
    halves when tensoring, as the duals of a rigid category do; the
    underlying category of an optic is symmetric, so the two halves tensor
    side by side and a gradient comes back in the order of the inputs.

    >>> x, y = Ty[int](1, 2), Ty[int](3, 4)
    >>> assert x @ y == Ty[int](1 + 3, 2 + 4)
    >>> assert -(x @ y) == -x @ -y
    """
    natural = monoidal.Ty

    def tensor(self, *others: Ty) -> Ty:
        if any(not isinstance(other, Ty) for other in others):
            return NotImplemented
        unit = type(self).natural()
        positive = sum([x.positive for x in (self, ) + others], unit)
        negative = sum([x.negative for x in (self, ) + others], unit)
        return type(self)(positive, negative)

    __matmul__ = __add__ = tensor

    def __repr__(self):
        pos, neg = repr(self.positive), repr(self.negative)
        return f"optics.Ty[{factory_name(self.natural)}]"\
               f"(positive={pos}, negative={neg})"

    def __str__(self):
        try:
            return " @ ".join(list(map(str, self.positive)) + [
                f"-{x}" for x in self.negative])
        except TypeError:  # e.g. when Ty.natural == int
            return repr(self)


def pairs(category) -> type:
    """ The pairs of objects of a category, e.g. of tuples of types. """
    return Ty[get_origin(category.ob) or category.ob]


@dataclass
class Optic(SymmetricCategory, NamedGeneric['category']):
    """
    An optic from `dom` to `cod` is a `residual` with a `forward` morphism
    `dom.positive -> residual @ cod.positive` and a `backward` morphism
    `residual @ cod.negative -> dom.negative` in an underlying `category`.

    Parameters:
        dom (Ty) : The domain of the optic.
        cod (Ty) : The codomain of the optic.
        forward (category) :
            The morphism ``dom.positive -> residual @ cod.positive``.
        backward (category) :
            The morphism ``residual @ cod.negative -> dom.negative``.
        residual (category.ob) : The residual, empty by default.

    .. admonition:: Summary

        .. autosummary::

            lift
            id
            then
            tensor
            swap
            to_int
            to_lens
    """
    category = symmetric.Diagram
    ob = classproperty(lambda cls: pairs(cls.category))

    dom: ob
    cod: ob
    forward: category
    backward: category
    residual: category.ob = None

    def __post_init__(self):
        if self.residual is None:
            self.residual = self.category.ob()
        assert_isinstance(self.dom, Ty)
        assert_isinstance(self.cod, Ty)
        assert_isinstance(self.forward, self.category)
        assert_isinstance(self.backward, self.category)
        identity = self.category.id
        assert_iscomposable(identity(self.dom.positive), self.forward)
        assert_iscomposable(
            self.forward, identity(self.residual + self.cod.positive))
        assert_iscomposable(
            identity(self.residual + self.cod.negative), self.backward)
        assert_iscomposable(self.backward, identity(self.dom.negative))

    def __repr__(self):
        return factory_name(type(self)) + f"({self.dom!r}, {self.cod!r}, "\
            f"{self.forward!r}, {self.backward!r}, {self.residual!r})"

    @classmethod
    def lift(cls, forward: category, backward: category = None) -> Optic:
        """
        A morphism `x -> y` of the underlying category as an optic from
        `(x, x_)` to `(y, y_)` with empty residual, the backward morphism
        `y_ -> x_` being the identity on the unit by default: the two
        injection functors from the category and its opposite.

        Parameters:
            forward : The morphism to lift on the positive halves.
            backward : The morphism to lift on the negative halves.
        """
        if backward is None:
            backward = cls.category.id(cls.category.ob())
        return cls(cls.ob(forward.dom, backward.cod),
                   cls.ob(forward.cod, backward.dom), forward, backward)

    @classmethod
    def id(cls, dom: ob = None) -> Optic:
        """
        The identity optic on `dom`, with empty residual.

        Parameters:
            dom : The domain of the identity, also its codomain.
        """
        dom = cls.ob() if dom is None else dom
        return cls.lift(
            cls.category.id(dom.positive), cls.category.id(dom.negative))

    @unbiased
    def then(self, other: Optic) -> Optic:
        """
        Sequential composition tensors the residuals: the forward legs
        compose past the first residual, the backward legs in reverse.

        Parameters:
            other : The optic to compose with.
        """
        assert_iscomposable(self, other)
        residual = self.category.id(self.residual)
        forward = self.forward >> residual @ other.forward
        backward = residual @ other.backward >> self.backward
        return type(self)(self.dom, other.cod, forward, backward,
                          self.residual + other.residual)

    @unbiased
    def tensor(self, other: Optic) -> Optic:
        """
        Parallel composition tensors the residuals, swapping the residual
        of `other` past the codomain of `self` on both legs.

        Parameters:
            other : The optic to compose in parallel.
        """
        identity, swap = self.category.id, self.category.swap
        forward = self.forward @ other.forward >> identity(self.residual)\
            @ swap(self.cod.positive, other.residual)\
            @ identity(other.cod.positive)
        backward = identity(self.residual)\
            @ swap(other.residual, self.cod.negative)\
            @ identity(other.cod.negative) >> self.backward @ other.backward
        return type(self)(self.dom @ other.dom, self.cod @ other.cod,
                          forward, backward, self.residual + other.residual)

    @classmethod
    def swap(cls, left: ob, right: ob) -> Optic:
        """
        The swap of two pairs is the swap of their positive halves forward
        and of their negative halves backward, with empty residual.

        Parameters:
            left : The pair on the left of the swap.
            right : The pair on the right of the swap.
        """
        return cls.lift(cls.category.swap(left.positive, right.positive),
                        cls.category.swap(right.negative, left.negative))

    def to_int(self) -> interaction.Diagram:
        """
        The integer diagram `dom.positive @ cod.negative -> cod.positive @
        dom.negative` of an optic over a traced category: the residual is
        the wire between the two legs, so that composition of optics is the
        symmetric feedback of :mod:`discopy.interaction` up to the axioms of
        traced categories, and the tensor is that of integer diagrams up to
        the swap of the negative halves.

        >>> from discopy.symmetric import Ty as T, Box
        >>> x, x_, y, y_, m = map(T, ["x", "x'", "y", "y'", "m"])
        >>> f = Optic(Ty(x, x_), Ty(y, y_),
        ...           Box('f', x, m @ y), Box("f'", m @ y_, x_), m)
        >>> f.to_int().draw(doctest="docs/_static/optics/to-int.svg")

        .. image:: /_static/optics/to-int.svg
            :align: center
        """
        identity, swap = self.category.id, self.category.swap
        inside = self.forward @ identity(self.cod.negative)\
            >> swap(self.residual, self.cod.positive)\
            @ identity(self.cod.negative)\
            >> identity(self.cod.positive) @ self.backward
        ob = interaction.Ty[self.ob.natural]
        return interaction.Diagram[self.category](
            inside, ob(*self.dom), ob(*self.cod))

    def to_lens(self) -> Lens:
        """
        The lens of an optic over a Markov category: `get` discards the
        residual, `put` recomputes it from the input and discards the
        output, so that the residual is `dom.positive` up to sliding.
        """
        identity, discard = self.category.id, self.category.discard
        positive, negative, residual = *self.cod, self.residual
        get = self.forward >> discard(residual) @ identity(positive)
        put = self.forward @ identity(negative) >> identity(residual)\
            @ discard(positive) @ identity(negative) >> self.backward
        return Lens[self.category](self.dom, self.cod, get, put)


@dataclass
class Lens(SymmetricCategory, NamedGeneric['category']):
    """
    A lens from `dom` to `cod` is a morphism `get : dom.positive ->
    cod.positive` and a morphism `put : dom.positive @ cod.negative ->
    dom.negative` in an underlying Markov `category`.

    Parameters:
        dom (Ty) : The domain of the lens.
        cod (Ty) : The codomain of the lens.
        get (category) : The morphism ``dom.positive -> cod.positive``.
        put (category) :
            The morphism ``dom.positive @ cod.negative -> dom.negative``.

    Note
    ----
    Lenses are the optics whose residual is the input itself, copied. The
    identity is a unit for composition up to the counit law on the left and
    the naturality of discard on the right, i.e. the axioms of Markov
    categories; the lens laws hold when `get` and `put` are the projection
    and update of a cartesian product, i.e. when copy is natural for both.
    Lenses form a symmetric category and not a Markov one: copying a pair
    would ask for a monoid on its negative half, which is how the reverse
    derivative of a fan-out sums the gradients.

    .. admonition:: Summary

        .. autosummary::

            lift
            id
            then
            tensor
            swap
            to_optic
    """
    category = markov.Diagram
    ob = classproperty(lambda cls: pairs(cls.category))

    dom: ob
    cod: ob
    get: category
    put: category

    def __post_init__(self):
        assert_isinstance(self.dom, Ty)
        assert_isinstance(self.cod, Ty)
        assert_isinstance(self.get, self.category)
        assert_isinstance(self.put, self.category)
        identity = self.category.id
        assert_iscomposable(identity(self.dom.positive), self.get)
        assert_iscomposable(self.get, identity(self.cod.positive))
        assert_iscomposable(
            identity(self.dom.positive + self.cod.negative), self.put)
        assert_iscomposable(self.put, identity(self.dom.negative))

    def __repr__(self):
        return factory_name(type(self)) + f"({self.dom!r}, {self.cod!r}, "\
            f"{self.get!r}, {self.put!r})"

    @classmethod
    def lift(cls, get: category, backward: category = None) -> Lens:
        """
        A morphism `x -> y` of the underlying category as a lens from
        `(x, x_)` to `(y, y_)`, with `put` discarding the input and applying
        the `backward` morphism `y_ -> x_`, the identity on the unit by
        default.

        Parameters:
            get : The morphism to lift on the positive halves.
            backward : The morphism to lift on the negative halves.
        """
        if backward is None:
            backward = cls.category.id(cls.category.ob())
        return cls(cls.ob(get.dom, backward.cod),
                   cls.ob(get.cod, backward.dom),
                   get, cls.category.discard(get.dom) @ backward)

    @classmethod
    def id(cls, dom: ob = None) -> Lens:
        """
        The identity lens on `dom`: `get` is the identity and `put` discards
        the input.

        Parameters:
            dom : The domain of the identity, also its codomain.
        """
        dom = cls.ob() if dom is None else dom
        return cls.lift(
            cls.category.id(dom.positive), cls.category.id(dom.negative))

    @unbiased
    def then(self, other: Lens) -> Lens:
        """
        Sequential composition is the chain rule: `put` copies the input,
        reads it with `self.get`, writes with `other.put` then `self.put`.

        Parameters:
            other : The lens to compose with.
        """
        assert_iscomposable(self, other)
        identity, copy = self.category.id, self.category.copy
        positive, negative = self.dom.positive, other.cod.negative
        get = self.get >> other.get
        put = copy(positive) @ identity(negative)\
            >> identity(positive) @ self.get @ identity(negative)\
            >> identity(positive) @ other.put >> self.put
        return type(self)(self.dom, other.cod, get, put)

    @unbiased
    def tensor(self, other: Lens) -> Lens:
        """
        Parallel composition swaps the input of `other` past the output of
        `self` before applying both `put`.

        Parameters:
            other : The lens to compose in parallel.
        """
        identity, swap = self.category.id, self.category.swap
        get = self.get @ other.get
        put = identity(self.dom.positive)\
            @ swap(other.dom.positive, self.cod.negative)\
            @ identity(other.cod.negative) >> self.put @ other.put
        return type(self)(self.dom @ other.dom, self.cod @ other.cod, get, put)

    @classmethod
    def swap(cls, left: ob, right: ob) -> Lens:
        """
        The swap of two pairs is the swap of their positive halves as `get`
        and of their negative halves as `put`, discarding the input.

        Parameters:
            left : The pair on the left of the swap.
            right : The pair on the right of the swap.
        """
        return cls.lift(cls.category.swap(left.positive, right.positive),
                        cls.category.swap(right.negative, left.negative))

    def to_optic(self) -> Optic:
        """
        The optic of a lens: the residual is the input, copied to be read
        by `get` and kept for `put`.

        >>> from discopy.markov import Ty as T, Box
        >>> x, x_, y, y_ = map(T, ["x", "x'", "y", "y'"])
        >>> lens = Lens(Ty(x, x_), Ty(y, y_),
        ...             Box('get', x, y), Box('put', x @ y_, x_))
        >>> lens.to_optic().to_int().draw(
        ...     doctest="docs/_static/optics/lens.svg")

        .. image:: /_static/optics/lens.svg
            :align: center
        """
        identity, copy = self.category.id, self.category.copy
        positive = self.dom.positive
        forward = copy(positive) >> identity(positive) @ self.get
        return Optic[self.category](
            self.dom, self.cod, forward, self.put, positive)
