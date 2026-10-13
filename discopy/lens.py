# -*- coding: utf-8 -*-

"""
Monoidal lenses, also called optics, over an underlying `category`, with
one class for each doctrine as in :mod:`discopy.para`: :class:`Symmetric`
over a symmetric category, :class:`Traced` over a traced one and
:class:`Markov`, the lenses `(get, put)` of functional programming, over a
Markov one.

A lens from a pair `(x, x_)` to a pair `(y, y_)` is a residual `m` with a
`forward` morphism `x -> y @ m` and a `backward` morphism `m @ y_ -> x_` in
the underlying category, see :cite:t:`Riley18`, the residual on the right
of the forward leg and on the left of the backward one, so that the two
legs side by side draw as a comb. Composition tensors the residuals, as in
the category of parametric maps :mod:`discopy.para`: the forward leg is a
coparametric map with the residual as coparameter, the backward leg a
parametric map with the residual as parameter :cite:p:`CapucciEtAl21`.
When the underlying category is cartesian, a lens is a pair `(get, put)`
with the residual normalised to `x` by copying, the bidirectional accessors
of :cite:t:`ClarkeEtAl20`; when it is traced, a lens is an integer
diagram :mod:`discopy.interaction` with the residual as the wire between
the two legs. Two lenses are equal when their representatives are; the
quotient by sliding a morphism across the residual is decided by
:meth:`Symmetric.to_int` for diagrams in the free traced category and by
:meth:`Symmetric.to_markov` when copy is natural, i.e. in a cartesian
category: over a Markov category, `to_markov` forgets the correlation
between output and residual.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Symmetric
    Traced
    Markov

Example
-------

Lenses over :class:`Function <discopy.python.Function>` are the accessors of
functional programming: `get` reads a part of a structure and `put` writes
it back. The lens on the first component of a pair:

>>> from discopy import python
>>> from discopy.python import Function
>>> P = Ty[python.Ty]((int, str), (int, str))
>>> A = Ty[python.Ty]((int, ), (int, ))
>>> first = Markov[Function](P, A,
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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from discopy import interaction, markov, messages, monoidal, symmetric
from discopy.abc import (
    MarkovCategory, NamedGeneric, SymmetricCategory, TracedCategory)
from discopy.utils import (
    AxiomError, assert_iscomposable, assert_isinstance, classproperty,
    factory_name, get_origin, unbiased)


class Ty(interaction.Ty):
    """
    A pair of types of the underlying category, tensored componentwise.

    Parameters:
        positive : The forward half of the type.
        negative : The backward half of the type.

    Note
    ----
    An :class:`interaction.Ty <discopy.interaction.Ty>` reverses the
    negative halves when tensoring, as the duals of a rigid category do;
    the underlying category of a lens is symmetric, so its
    :attr:`negatives` stay side by side and a gradient comes back in the
    order of the inputs.

    >>> from discopy.monoidal import List
    >>> x = Ty[List[str]](("x", ), ("x'", ))
    >>> y = Ty[List[str]](("y", ), ("y'", ))
    >>> assert x @ y == Ty[List[str]](("x", "y"), ("x'", "y'"))
    >>> assert -(x @ y) == -x @ -y
    """
    natural = monoidal.Ty
    negatives = staticmethod(tuple)


def pairs(category) -> type:
    """ The pairs of objects of a category, e.g. of lists of types. """
    return Ty[category.ob]


@dataclass
class Symmetric(SymmetricCategory, NamedGeneric['category']):
    """
    A lens from `dom` to `cod` over a symmetric `category` is a `residual`
    with a `forward` morphism `dom.positive -> cod.positive @ residual` and
    a `backward` morphism `residual @ cod.negative -> dom.negative`.

    Parameters:
        dom (Ty) : The domain of the lens.
        cod (Ty) : The codomain of the lens.
        forward (category) :
            The morphism ``dom.positive -> cod.positive @ residual``.
        backward (category) :
            The morphism ``residual @ cod.negative -> dom.negative``.
        residual (category.ob) : The residual, empty by default.

    Note
    ----
    The structure of the underlying category lifts leg by leg, or not at
    all. Over a traced category, lenses are traced, see :class:`Traced`.
    Over a compact or hypergraph category, cups, caps and spiders lift with
    empty residual, since :meth:`lift` is a strict monoidal functor from
    the category and its opposite: the dual of `(x, x_)` is `(x.r, x_.r)`,
    with the cup of `x` and the cap of `x_` as legs, and the copy of
    `(x, x_)` has the copy of `x` and the merge of `x_` as legs, which is
    how the reverse derivative of a fan-out sums the gradients. Over a
    Markov category, lenses are only symmetric: copying a pair would ask
    for a merge on its negative half and discarding for a unit, which is
    why :class:`Markov` is a symmetric category too.

    .. admonition:: Summary

        .. autosummary::

            lift
            id
            then
            tensor
            swap
            to_int
            to_markov
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
        assert_isinstance(self.dom, self.ob)
        assert_isinstance(self.cod, self.ob)
        assert_isinstance(self.forward, self.category)
        assert_isinstance(self.backward, self.category)
        assert_isinstance(self.residual, self.category.ob)
        identity = self.category.id
        assert_iscomposable(identity(self.dom.positive), self.forward)
        assert_iscomposable(
            self.forward, identity(self.cod.positive @ self.residual))
        assert_iscomposable(
            identity(self.residual @ self.cod.negative), self.backward)
        assert_iscomposable(self.backward, identity(self.dom.negative))

    def __repr__(self):
        factory, category = map(
            factory_name, (get_origin(type(self)), self.category))
        return f"{factory}[{category}]({self.dom!r}, {self.cod!r}, "\
            f"{self.forward!r}, {self.backward!r}, {self.residual!r})"

    @classmethod
    def lift(cls, forward: category, backward: category = None) -> Symmetric:
        """
        A morphism `x -> y` of the underlying category as a lens from
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
    def id(cls, dom: ob = None) -> Symmetric:
        """
        The identity lens on `dom`, with empty residual.

        Parameters:
            dom : The domain of the identity, also its codomain.

        Example
        -------
        >>> from discopy.symmetric import Ty as T, Diagram
        >>> x, x_ = T('x'), T("x'")
        >>> X = Ty(x, x_)
        >>> assert Symmetric.id(X)\\
        ...     == Symmetric.lift(Diagram.id(x), Diagram.id(x_))
        """
        dom = cls.ob() if dom is None else dom
        return cls.lift(
            cls.category.id(dom.positive), cls.category.id(dom.negative))

    @unbiased
    def then(self, other: Symmetric) -> Symmetric:
        """
        Sequential composition tensors the residuals: the forward legs
        compose then swap the second residual past the first, the backward
        legs compose in reverse past the first residual, as
        :meth:`para.Symmetric.then <discopy.para.Symmetric.then>` does with its
        coparameters.

        Parameters:
            other : The lens to compose with.

        Example
        -------
        >>> from discopy.symmetric import Ty as T, Box, Diagram
        >>> x, x_, y, y_, z, z_ = map(T, ["x", "x'", "y", "y'", "z", "z'"])
        >>> m, n = map(T, "mn")
        >>> X, Y, Z = Ty(x, x_), Ty(y, y_), Ty(z, z_)
        >>> f = Symmetric(X, Y, Box('f', x, y @ m), Box("f'", m @ y_, x_), m)
        >>> g = Symmetric(Y, Z, Box('g', y, z @ n), Box("g'", n @ z_, y_), n)
        >>> assert (f >> g).residual == m @ n
        >>> assert (f >> g).forward\\
        ...     == f.forward >> g.forward @ m >> z @ Diagram.swap(n, m)
        >>> assert (f >> g).backward == m @ g.backward >> f.backward
        >>> (f >> g).to_int().draw(doctest="docs/_static/lens/then.svg")

        .. image:: /_static/lens/then.svg
            :align: center
        """
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        identity, swap = self.category.id, self.category.swap
        forward = self.forward >> other.forward @ identity(self.residual)\
            >> identity(other.cod.positive)\
            @ swap(other.residual, self.residual)
        backward = identity(self.residual) @ other.backward >> self.backward
        return type(self)(self.dom, other.cod, forward, backward,
                          self.residual @ other.residual)

    @unbiased
    def tensor(self, other: Symmetric) -> Symmetric:
        """
        Parallel composition tensors the residuals: the forward legs swap
        the residual of `self` past the output of `other`, the backward legs
        swap the residual of `other` past the input of `self`.

        Parameters:
            other : The lens to compose in parallel.

        Example
        -------
        >>> from discopy.symmetric import Ty as T, Box, Diagram
        >>> x, x_, y, y_, z, z_, w, w_ = map(
        ...     T, ["x", "x'", "y", "y'", "z", "z'", "w", "w'"])
        >>> m, k = map(T, "mk")
        >>> X, Y, Z, W = Ty(x, x_), Ty(y, y_), Ty(z, z_), Ty(w, w_)
        >>> f = Symmetric(X, Y, Box('f', x, y @ m), Box("f'", m @ y_, x_), m)
        >>> h = Symmetric(Z, W, Box('h', z, w @ k), Box("h'", k @ w_, z_), k)
        >>> assert (f @ h).residual == m @ k
        >>> assert (f @ h).dom == X @ Z and (f @ h).cod == Y @ W
        >>> assert (f @ h).forward\\
        ...     == f.forward @ h.forward >> y @ Diagram.swap(m, w) @ k
        >>> assert (f @ h).backward\\
        ...     == m @ Diagram.swap(k, y_) @ w_ >> f.backward @ h.backward
        >>> (f @ h).to_int().draw(doctest="docs/_static/lens/tensor.svg")

        .. image:: /_static/lens/tensor.svg
            :align: center
        """
        assert_isinstance(other, type(self))
        identity, swap = self.category.id, self.category.swap
        forward = self.forward @ other.forward >> identity(self.cod.positive)\
            @ swap(self.residual, other.cod.positive)\
            @ identity(other.residual)
        backward = identity(self.residual)\
            @ swap(other.residual, self.cod.negative)\
            @ identity(other.cod.negative) >> self.backward @ other.backward
        return type(self)(self.dom @ other.dom, self.cod @ other.cod,
                          forward, backward, self.residual @ other.residual)

    @classmethod
    def swap(cls, left: ob, right: ob) -> Symmetric:
        """
        The swap of two pairs is the swap of their positive halves forward
        and of their negative halves backward, with empty residual.

        Parameters:
            left : The pair on the left of the swap.
            right : The pair on the right of the swap.

        Example
        -------
        >>> from discopy.symmetric import Ty as T, Diagram
        >>> x, x_, y, y_ = map(T, ["x", "x'", "y", "y'"])
        >>> X, Y = Ty(x, x_), Ty(y, y_)
        >>> assert Symmetric.swap(X, Y).forward == Diagram.swap(x, y)
        >>> assert Symmetric.swap(X, Y).backward == Diagram.swap(y_, x_)
        """
        return cls.lift(cls.category.swap(left.positive, right.positive),
                        cls.category.swap(right.negative, left.negative))

    def to_int(self) -> interaction.Diagram:
        """
        The integer diagram `dom.positive @ cod.negative -> cod.positive @
        dom.negative` of a lens over a traced category: the two legs side
        by side with the residual as the wire between them, no swap, so that
        composition of lenses is the symmetric feedback of
        :mod:`discopy.interaction` up to the axioms of traced categories,
        and the tensor is that of integer diagrams up to the swap of the
        negative halves.

        >>> from discopy.symmetric import Ty as T, Box
        >>> x, x_, y, y_, m = map(T, ["x", "x'", "y", "y'", "m"])
        >>> f = Symmetric(Ty(x, x_), Ty(y, y_),
        ...           Box('f', x, y @ m), Box("f'", m @ y_, x_), m)
        >>> f.to_int().draw(doctest="docs/_static/lens/to-int.svg")

        .. image:: /_static/lens/to-int.svg
            :align: center
        """
        inside = self.forward @ self.cod.negative\
            >> self.cod.positive @ self.backward
        ob = interaction.Ty[self.ob.natural]
        return interaction.Diagram[self.category](
            inside, ob(*self.dom), ob(*self.cod))

    def to_markov(self, discard: Callable = None) -> Markov:
        """
        The lens `(get, put)`: `get` discards the residual, `put` recomputes
        it from the input and discards the output. When copy is natural,
        i.e. over a cartesian category, this decides the quotient of lenses
        by sliding a morphism across the residual, which is then
        `dom.positive` up to sliding; over a Markov category, it forgets the
        correlation between output and residual.

        Parameters:
            discard : The discard of the underlying category by default, or
                an explicit one over a category without: the lens then has
                a `put` but no composition, which copies.
        """
        markov_lens = Markov[self.category]
        if discard is None:
            markov_lens.assert_ismarkov()
            discard = self.category.discard
        identity = self.category.id
        positive, negative, residual = *self.cod, self.residual
        get = self.forward >> identity(positive) @ discard(residual)
        put = self.forward @ identity(negative) >> discard(positive)\
            @ identity(residual @ negative) >> self.backward
        return markov_lens(self.dom, self.cod, get, put)


class Traced(Symmetric, TracedCategory):
    """
    Lenses over a traced category are traced leg by leg: the forward leg
    over the positive halves, with the residual swapped out of the way, the
    backward leg over the negative halves, so that :meth:`Symmetric.to_int`
    is a traced functor into :mod:`discopy.interaction`.

    Example
    -------
    >>> from discopy.symmetric import Ty as T, Box, Diagram
    >>> x, x_, u, u_, m = map(T, ["x", "x'", "u", "u'", "m"])
    >>> X, U = Ty(x, x_), Ty(u, u_)
    >>> f = Traced(X @ U, X @ U, Box('f', x @ u, x @ u @ m),
    ...            Box("f'", m @ x_ @ u_, x_ @ u_), m)
    >>> assert f.trace().dom == X == f.trace().cod
    >>> assert f.trace().forward\\
    ...     == (f.forward >> x @ Diagram.swap(u, m)).trace()
    >>> assert f.trace().backward == f.backward.trace()
    """
    def trace(self, n: int = 1, left: bool = False) -> Traced:
        """
        The trace of a lens over the last `n` atoms of each half of its
        domain and codomain, the residual never being traced.

        Parameters:
            n : The number of atoms to trace over on each half.
            left : Whether to trace on the left, not implemented.
        """
        if n == 0:
            return self
        if left:
            raise NotImplementedError
        identity, swap = self.category.id, self.category.swap
        (x, x_), (y, y_) = (
            (positive[:-n], negative[:-n])
            for positive, negative in (self.dom, self.cod))
        u = self.dom.positive[-n:]
        forward = self.forward >> identity(y) @ swap(u, self.residual)
        return type(self)(self.ob(x, x_), self.ob(y, y_), forward.trace(n),
                          self.backward.trace(n), self.residual)


@dataclass
class Markov(SymmetricCategory, NamedGeneric['category']):
    """
    A lens from `dom` to `cod` over a Markov `category` is a morphism
    `get : dom.positive -> cod.positive` and a morphism
    `put : dom.positive @ cod.negative -> dom.negative`.

    Parameters:
        dom (Ty) : The domain of the lens.
        cod (Ty) : The codomain of the lens.
        get (category) : The morphism ``dom.positive -> cod.positive``.
        put (category) :
            The morphism ``dom.positive @ cod.negative -> dom.negative``.

    Note
    ----
    These are the lenses whose residual is the input itself, copied. The
    identity is a unit for composition up to the counit law on the left and
    the naturality of discard on the right, i.e. the axioms of Markov
    categories; composition is associative up to the naturality of copy for
    `get`, i.e. when `get` is deterministic, as every morphism of a
    cartesian category is. They form a symmetric category and not a Markov
    one: copying a pair would ask for a monoid on its negative half, which is
    how the reverse derivative of a fan-out sums the gradients. A neural
    network is a parametric lens, i.e. a
    :class:`para.Symmetric <discopy.para.Symmetric>` over `lens.Markov`; see
    :mod:`discopy.neural`.

    .. admonition:: Summary

        .. autosummary::

            lift
            id
            then
            tensor
            swap
            to_symmetric
    """
    category = markov.Diagram
    ob = classproperty(lambda cls: pairs(cls.category))

    dom: ob
    cod: ob
    get: category
    put: category

    def __post_init__(self):
        assert_isinstance(self.dom, self.ob)
        assert_isinstance(self.cod, self.ob)
        assert_isinstance(self.get, self.category)
        assert_isinstance(self.put, self.category)
        identity = self.category.id
        assert_iscomposable(identity(self.dom.positive), self.get)
        assert_iscomposable(self.get, identity(self.cod.positive))
        assert_iscomposable(
            identity(self.dom.positive @ self.cod.negative), self.put)
        assert_iscomposable(self.put, identity(self.dom.negative))

    def __repr__(self):
        factory, category = map(
            factory_name, (get_origin(type(self)), self.category))
        return f"{factory}[{category}]"\
            f"({self.dom!r}, {self.cod!r}, {self.get!r}, {self.put!r})"

    @classmethod
    def assert_ismarkov(cls):
        """ Assert that :attr:`category` has copy and discard. """
        if not issubclass(cls.category, MarkovCategory):
            raise AxiomError(messages.NOT_MARKOV.format(
                factory_name(cls.category)))

    @classmethod
    def lift(cls, get: category, backward: category = None) -> Markov:
        """
        A morphism `x -> y` of the underlying category as a lens from
        `(x, x_)` to `(y, y_)`, with `put` discarding the input and applying
        the `backward` morphism `y_ -> x_`, the identity on the unit by
        default.

        Parameters:
            get : The morphism to lift on the positive halves.
            backward : The morphism to lift on the negative halves.
        """
        cls.assert_ismarkov()
        if backward is None:
            backward = cls.category.id(cls.category.ob())
        return cls(cls.ob(get.dom, backward.cod),
                   cls.ob(get.cod, backward.dom),
                   get, cls.category.discard(get.dom) @ backward)

    @classmethod
    def id(cls, dom: ob = None) -> Markov:
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
    def then(self, other: Markov) -> Markov:
        """
        Sequential composition is the chain rule: `put` copies the input,
        reads it with `self.get`, writes with `other.put` then `self.put`.
        This is why lenses over :class:`Function <discopy.python.Function>`
        are the semantics of reverse-mode differentiation
        :cite:p:`CruttwellEtAl22`: the reverse derivative of a function `f`
        is the lens with `get` its value and `put` its Jacobian transposed,
        applied to the incoming gradient.

        Parameters:
            other : The lens to compose with.

        Example
        -------
        >>> from discopy import python
        >>> from discopy.python import Function
        >>> R = Ty[python.Ty]((float, ), (float, ))
        >>> square = Markov[Function](R, R,
        ...     Function(lambda x: x * x, (float, ), (float, )),
        ...     Function(lambda x, dy: 2 * x * dy, (float, float), (float, )))
        >>> (square >> square).get(3.)
        81.0
        >>> (square >> square).put(3., 1.)
        108.0
        """
        assert_isinstance(other, type(self))
        self.assert_ismarkov()
        assert_iscomposable(self, other)
        identity, copy = self.category.id, self.category.copy
        positive, negative = self.dom.positive, other.cod.negative
        get = self.get >> other.get
        put = copy(positive) @ identity(negative)\
            >> identity(positive) @ self.get @ identity(negative)\
            >> identity(positive) @ other.put >> self.put
        return type(self)(self.dom, other.cod, get, put)

    @unbiased
    def tensor(self, other: Markov) -> Markov:
        """
        Parallel composition swaps the input of `other` past the output of
        `self` before applying both `put`.

        Parameters:
            other : The lens to compose in parallel.
        """
        assert_isinstance(other, type(self))
        identity, swap = self.category.id, self.category.swap
        get = self.get @ other.get
        put = identity(self.dom.positive)\
            @ swap(other.dom.positive, self.cod.negative)\
            @ identity(other.cod.negative) >> self.put @ other.put
        return type(self)(self.dom @ other.dom, self.cod @ other.cod, get, put)

    @classmethod
    def swap(cls, left: ob, right: ob) -> Markov:
        """
        The swap of two pairs is the swap of their positive halves as `get`
        and of their negative halves as `put`, discarding the input.

        Parameters:
            left : The pair on the left of the swap.
            right : The pair on the right of the swap.
        """
        return cls.lift(cls.category.swap(left.positive, right.positive),
                        cls.category.swap(right.negative, left.negative))

    def to_symmetric(self) -> Symmetric:
        """
        The lens over the underlying symmetric category: the residual is the
        input, copied to be read by `get` and kept for `put`.

        >>> from discopy.markov import Ty as T, Box
        >>> x, x_, y, y_ = map(T, ["x", "x'", "y", "y'"])
        >>> f = Markov(Ty(x, x_), Ty(y, y_),
        ...            Box('get', x, y), Box('put', x @ y_, x_))
        >>> f.to_symmetric().to_int().draw(
        ...     doctest="docs/_static/lens/markov.svg")

        .. image:: /_static/lens/markov.svg
            :align: center
        """
        self.assert_ismarkov()
        identity, copy = self.category.id, self.category.copy
        positive = self.dom.positive
        forward = copy(positive) >> self.get @ identity(positive)
        return Symmetric[self.category](
            self.dom, self.cod, forward, self.put, positive)
