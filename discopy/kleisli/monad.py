# -*- coding: utf-8 -*-

"""
Monads on the category of Python functions, as monoids in the category of
endofunctors.

A :class:`Monad` is an :class:`EndoFunctor` ``M`` built from value-level
callables ``ob, fmap, pure, join``, with the monoid structure exposed as two
:class:`Transformation`, the :attr:`Monad.unit` ``id -> M`` and the
:attr:`Monad.mult` ``M >> M -> M``.

Sub-additive monads also come with ``zero, plus, mass, support`` which the
additive fragment of :mod:`discopy.kleisli` uses to compute traces, i.e.
while loops, see :meth:`discopy.kleisli.additive.Channel.trace`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    EndoFunctor
    Transformation
    Monad

.. admonition:: Instances

    .. autosummary::
        :nosignatures:
        :toctree:

        Maybe
        Powerset
        Subdistribution
        Writer

Example
-------
>>> assert Maybe.bind(Maybe.pure(42), lambda a: Maybe.pure(a + 1)) == 43
>>> assert Powerset.bind(
...     frozenset({1, 2}), lambda a: Powerset.pure(a % 2)) == {0, 1}
"""

from __future__ import annotations

from collections.abc import Callable

from discopy import cat
from discopy.python import function
from discopy.utils import assert_isinstance, tuplify


class EndoFunctor(cat.Functor):
    """
    An endofunctor on the category of Python functions, i.e. a pair of maps
    ``ob`` from types to types and ``ar`` from functions to functions.

    Parameters:
        ob : Map from tuples of types to tuples of types.
        ar : Map from :class:`discopy.python.function.Function` to itself.

    Example
    -------
    >>> F = EndoFunctor(
    ...     lambda ty: (list, ),
    ...     lambda f: function.Function(
    ...         lambda xs: list(map(f, xs)), (list, ), (list, )))
    >>> assert F(int) == (list, )
    >>> succ = function.Function(lambda n: n + 1, int, int)
    >>> assert F(succ)([1, 2, 3]) == [2, 3, 4]
    """
    dom = cod = function.Function

    __hash__ = lambda self: id(self)

    def __call__(self, other):
        if isinstance(other, function.Function):
            return self.ar_map[other]
        return tuplify(self.ob_map[tuplify(other)])

    def then(self, other: EndoFunctor) -> EndoFunctor:
        """
        The composition of an endofunctor with another, called with ``>>``.

        Parameters:
            other : The other endofunctor, to be applied second.
        """
        assert_isinstance(other, cat.Functor)
        return EndoFunctor(lambda x: other(self(x)), lambda f: other(self(f)))


class Transformation(cat.Transformation):
    """
    A transformation between two parallel :class:`EndoFunctor`, i.e. a mapping
    from tuples of types ``x`` to functions ``dom(x) -> cod(x)``.

    Parameters:
        components : Map from tuples of types to functions.
        dom : The domain endofunctor.
        cod : The codomain endofunctor.

    Example
    -------
    >>> assert Maybe.unit((int, ))(42) == 42
    >>> assert Maybe.mult((int, ))(Nothing) is Nothing
    """
    ob = EndoFunctor


class Monad(EndoFunctor):
    """
    A monad is an :class:`EndoFunctor` built from value-level callables,
    i.e. a monoid in the category of endofunctors with :attr:`unit` and
    :attr:`mult` as its two natural transformations.

    Parameters:
        name : The name of the monad, e.g. ``"Maybe"``.
        ob : The type constructor, a map from tuples of types to tuples of
            types.
        fmap : The functorial action ``fmap(g, ma)`` of a callable ``g`` on a
            monadic value ``ma``.
        pure : The unit ``pure(a)`` of the monad on a value ``a``.
        join : The multiplication ``join(mma)`` on a doubly-monadic value.
        bind : Optional override for ``bind(ma, f) = join(fmap(f, ma))``.
        zero : Optional, the empty monadic value for sub-additive monads.
        plus : Optional, the sum of two monadic values.
        mass : Optional, the total mass ``mass(ma) >= 0`` of a monadic value.
        support : Optional, the iterable of elements of a monadic value.
        filter_repeats : Whether the monad is idempotent enough that repeated
            elements can be dropped when computing additive traces.
        allclose : Approximate equality of monadic values, ``==`` by default.
        commutative : Whether the monad is commutative, i.e. whether its
            Kleisli category is monoidal rather than merely premonoidal.

    Example
    -------
    >>> assert Maybe.bind(21, lambda a: Maybe.pure(2 * a)) == 42
    >>> assert Maybe.bind(Nothing, lambda a: Maybe.pure(2 * a)) is Nothing
    """
    def __init__(
            self, name: str,
            ob: Callable, fmap: Callable, pure: Callable, join: Callable,
            bind: Callable = None,
            zero: Callable = None, plus: Callable = None,
            mass: Callable = None, support: Callable = None,
            filter_repeats: bool = False,
            allclose: Callable = None, commutative: bool = False):
        self.__name__ = self.name = name
        self.fmap, self.pure, self.join = fmap, pure, join
        self._bind = bind
        self.zero, self.plus, self.mass = zero, plus, mass
        self.support = support
        self.filter_repeats = filter_repeats
        self._allclose = allclose
        self.commutative = commutative
        super().__init__(
            ob=lambda ty: ob(ty),
            ar=lambda f: function.Function(
                lambda ma: fmap(f, ma), self(f.dom), self(f.cod)))

    def bind(self, ma, f: Callable):
        """
        The Kleisli extension ``bind(ma, f) = join(fmap(f, ma))``.

        Parameters:
            ma : A monadic value.
            f : A callable from values to monadic values.
        """
        if self._bind is not None:
            return self._bind(ma, f)
        return self.join(self.fmap(f, ma))

    def allclose(self, ma, mb, tol: float = 1e-12) -> bool:
        """
        Approximate equality of two monadic values, exact ``==`` by default.

        Parameters:
            ma : A monadic value.
            mb : Another monadic value.
            tol : The tolerance up to which to compare.
        """
        if self._allclose is not None:
            return self._allclose(ma, mb, tol)
        return ma == mb

    @property
    def is_additive(self) -> bool:
        """ Whether the monad has the sub-additive structure. """
        return None not in (
            self.zero, self.plus, self.mass, self.support)

    @property
    def unit(self) -> Transformation:
        """ The unit ``id -> M`` of the monad, with ``pure`` as components."""
        return Transformation(
            lambda x: function.Function(self.pure, tuplify(x), self(x)),
            EndoFunctor.id(), self)

    @property
    def mult(self) -> Transformation:
        """ The multiplication ``M >> M -> M``, with ``join`` as components."""
        square = self >> self
        return Transformation(
            lambda x: function.Function(self.join, square(x), self(x)),
            square, self)

    def double_strength(self, ma, mb, left: bool = True):
        """
        The double strength ``M(a) @ M(b) -> M(a @ b)``, running the effect of
        ``ma`` first if ``left`` else that of ``mb``.

        The left and right double strengths agree on all inputs if and only if
        the monad is commutative.

        Parameters:
            ma : A monadic value.
            mb : Another monadic value.
            left : Whether to run the effect of ``ma`` or ``mb`` first.

        Example
        -------
        >>> ma, mb = frozenset({1, 2}), frozenset({3})
        >>> assert Powerset.double_strength(ma, mb)\\
        ...     == Powerset.double_strength(ma, mb, left=False)\\
        ...     == {(1, 3), (2, 3)}
        """
        if left:
            return self.bind(
                ma, lambda a: self.fmap(lambda b: (a, b), mb))
        return self.bind(
            mb, lambda b: self.fmap(lambda a: (a, b), ma))

    def __repr__(self):
        return self.name


class NothingType:
    """ The type of the :obj:`Nothing` singleton. """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Nothing"


""" The unique inhabitant of :class:`NothingType`, i.e. the absent value. """
Nothing = NothingType()


def _maybe_plus(ma, mb):
    if ma is Nothing:
        return mb
    if mb is Nothing:
        return ma
    raise ValueError(f"Maybe.plus is partial: got {ma} and {mb}.")


""" The maybe monad, i.e. partial computation with :obj:`Nothing`. """
Maybe = Monad(
    "Maybe",
    ob=lambda ty: (object, ),
    fmap=lambda g, ma: Nothing if ma is Nothing else g(ma),
    pure=lambda a: a,
    join=lambda mma: mma,
    zero=lambda: Nothing,
    plus=_maybe_plus,
    mass=lambda ma: 0. if ma is Nothing else 1.,
    support=lambda ma: () if ma is Nothing else (ma, ),
    filter_repeats=True,
    commutative=True)


""" The finite powerset monad, i.e. non-deterministic computation. """
Powerset = Monad(
    "Powerset",
    ob=lambda ty: (frozenset, ),
    fmap=lambda g, ma: frozenset(map(g, ma)),
    pure=lambda a: frozenset({a}),
    join=lambda mma: frozenset().union(*mma),
    zero=frozenset,
    plus=lambda ma, mb: ma | mb,
    mass=lambda ma: float(len(ma)),
    support=lambda ma: ma,
    filter_repeats=True,
    commutative=True)


def _freeze(x):
    """ A hashable canonical form for dict-valued elements. """
    return frozenset(x.items()) if isinstance(x, dict) else x


def _thaw(x):
    return dict(x) if isinstance(x, frozenset) else x


def _subdist_fmap(g, ma: dict) -> dict:
    result = {}
    for a, p in ma.items():
        b = _freeze(g(a))
        result[b] = result.get(b, 0.) + p
    return result


def _subdist_join(mma: dict) -> dict:
    result = {}
    for inner, p in mma.items():
        for b, q in _thaw(inner).items():
            result[b] = result.get(b, 0.) + p * q
    return result


def _subdist_plus(ma: dict, mb: dict) -> dict:
    result = dict(ma)
    for b, q in mb.items():
        result[b] = result.get(b, 0.) + q
    return result


def _subdist_allclose(ma: dict, mb: dict, tol: float) -> bool:
    return all(
        abs(ma.get(a, 0.) - mb.get(a, 0.)) <= tol
        for a in set(ma) | set(mb))


""" The subdistribution monad, i.e. probabilistic computation with mass
at most one. """
Subdistribution = Monad(
    "Subdistribution",
    ob=lambda ty: (dict, ),
    fmap=_subdist_fmap,
    pure=lambda a: {a: 1.},
    join=_subdist_join,
    zero=dict,
    plus=_subdist_plus,
    mass=lambda ma: sum(ma.values()),
    support=lambda ma: ma.keys(),
    filter_repeats=False,
    allclose=_subdist_allclose,
    commutative=True)


""" The writer monad over the monoid of strings, i.e. logging.
It is not commutative, so its Kleisli category is merely premonoidal. """
Writer = Monad(
    "Writer",
    ob=lambda ty: (tuple, ),
    fmap=lambda g, ma: (ma[0], g(ma[1])),
    pure=lambda a: ("", a),
    join=lambda mma: (mma[0] + mma[1][0], mma[1][1]),
    commutative=False)
