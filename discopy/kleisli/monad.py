# -*- coding: utf-8 -*-

"""
Monads as monoids in the category of Python endofunctors.

A monad is an :class:`~discopy.python.function.EndoFunctor` ``M`` together
with a ``unit`` and a ``mult`` natural :class:`~discopy.cat.Transformation`

.. math::
    \\eta : \\mathrm{Id} \\Rightarrow M
    \\qquad\\qquad
    \\mu : M \\circ M \\Rightarrow M

satisfying the unit and associativity laws of a monoid, i.e. for every type
``X``

.. math::
    \\mu_X \\circ \\eta_{M(X)} = \\mathrm{id}_{M(X)} = \\mu_X \\circ M(\\eta_X)
    \\qquad\\qquad
    \\mu_X \\circ \\mu_{M(X)} = \\mu_X \\circ M(\\mu_X)

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Monad
    Maybe
    Powerset
    Subdistribution
    make_monad
"""
from __future__ import annotations

from collections.abc import Iterable

from discopy.cat import Transformation
from discopy.python.function import EndoFunctor, Function
from discopy.utils import assert_isinstance, tuplify, untuplify


class Monad:
    """
    A monad is a monoid in the category of Python endofunctors, i.e. an
    :class:`EndoFunctor` ``functor`` together with a ``unit`` and a ``mult``
    natural transformation satisfying the laws of a monoid.

    Parameters:
        name : The name of the monad, used e.g. by :class:`Channel[M]
            <discopy.kleisli.channel.Channel>` to name the Kleisli category.
        functor : The underlying endofunctor, i.e. the monad itself.
        unit : The natural transformation ``eta : Id -> functor``.
        mult : The natural transformation ``mu : functor >> functor
            -> functor``.

    Example
    -------
    The left and right unit laws hold for the :attr:`Maybe` monad on the
    type ``int``:

    >>> unit, mult = Maybe.unit, Maybe.mult
    >>> Mx = Maybe(int)
    >>> assert mult(int)(unit(Mx)(5)) == 5 == mult(int)(
    ...     Maybe.functor(unit(int))(5))

    The associativity law holds for the :attr:`Powerset` monad on ``int``:

    >>> mx = frozenset({frozenset({1, 2}), frozenset({3})})
    >>> lhs = Powerset.mult(int)(Powerset.mult(Powerset(int))(
    ...     frozenset({mx})))
    >>> rhs = Powerset.mult(int)(Powerset.functor(Powerset.mult(int))(
    ...     frozenset({mx})))
    >>> assert lhs == rhs == frozenset({1, 2, 3})
    """
    def __init__(
            self, name: str, functor: EndoFunctor,
            unit: Transformation, mult: Transformation):
        assert_isinstance(functor, EndoFunctor)
        assert_isinstance(unit, Transformation)
        assert_isinstance(mult, Transformation)
        self.__name__ = self.name = name
        self.functor, self.unit, self.mult = functor, unit, mult

    def __call__(self, X: type) -> tuple[type, ...]:
        """
        The type ``M(X)`` of computations with values in ``X`` and effects
        given by the monad.

        Parameters:
            X : The type of values.
        """
        return self.functor(X)

    @classmethod
    def from_maps(
            cls, name, ob_map, lift, unit_map, mult_map) -> Monad:
        """
        Build a monad from mappings on types and functions.

        Parameters:
            name : The name of the monad.
            ob_map : Mapping from a type ``X`` to the type ``M(X)``.
            lift : Mapping from a function ``f : X -> Y`` to ``M(f) : M(X)
                -> M(Y)``, i.e. the functorial action on functions.
            unit_map : Mapping from a type ``X`` to the function ``eta_X : X
                -> M(X)``.
            mult_map : Mapping from a type ``X`` to the function
                ``mu_X : M(M(X)) -> M(X)``.
        """
        unwrap = lambda X: untuplify(tuplify(X))
        functor = EndoFunctor(lambda X: (ob_map(untuplify(X)), ), lift)
        unit = Transformation(
            lambda X: unit_map(unwrap(X)), EndoFunctor.id(), functor)
        mult = Transformation(
            lambda X: mult_map(unwrap(X)), functor.then(functor), functor)
        return cls(name, functor, unit, mult)

    def __repr__(self):
        return f"Monad({self.name!r})"

    def __str__(self):
        return self.name


def make_monad(name, ob_map, lift, unit_map, mult_map) -> Monad:
    """
    Alias for :meth:`Monad.from_maps`.

    Parameters:
        name : The name of the monad.
        ob_map : Mapping from a type ``X`` to the type ``M(X)``.
        lift : Mapping from a function ``f : X -> Y`` to ``M(f) : M(X)
            -> M(Y)``, i.e. the functorial action on functions.
        unit_map : Mapping from a type ``X`` to the function ``eta_X : X
            -> M(X)``.
        mult_map : Mapping from a type ``X`` to the function ``mu_X : M(M(X))
            -> M(X)``.
    """
    return Monad.from_maps(name, ob_map, lift, unit_map, mult_map)


Maybe = Monad.from_maps(
    "Maybe",
    ob_map=lambda X: X | None,
    lift=lambda f: Function(
        lambda x: None if x is None else f(x),
        untuplify(f.dom) | None, untuplify(f.cod) | None),
    unit_map=lambda X: Function(lambda x: x, X, X | None),
    mult_map=lambda X: Function(lambda x: x, (X | None) | None, X | None))
"""
The maybe monad, sending a type ``X`` to ``X | None``: the unit and the
multiplication are both the identity, since Python's native optional type
does not distinguish ``None`` from a doubly-wrapped ``None``.
"""

Powerset = Monad.from_maps(
    "Powerset",
    ob_map=lambda X: frozenset[X],
    lift=lambda f: Function(
        lambda xs: frozenset(map(f, xs)),
        frozenset[untuplify(f.dom)], frozenset[untuplify(f.cod)]),
    unit_map=lambda X: Function(
        lambda x: frozenset({x}), X, frozenset[X]),
    mult_map=lambda X: Function(
        lambda xss: frozenset().union(*xss),
        frozenset[frozenset[X]], frozenset[X]))
"""
The powerset monad, sending a type ``X`` to ``frozenset[X]``: the unit takes
a singleton and the multiplication takes a union.
"""


def merge(pairs: Iterable) -> frozenset:
    """
    Sum the weights of duplicate outcomes in an iterator of pairs, used to
    build the :attr:`Subdistribution` monad's functor and multiplication.

    Parameters:
        pairs : An iterator of pairs of an outcome and its weight.
    """
    weights = {}
    for outcome, weight in pairs:
        weights[outcome] = weights.get(outcome, 0.) + weight
    return frozenset(weights.items())


def dist(X: type) -> type:
    """
    The representation type for subdistributions over ``X``, i.e. finite
    sets of pairs of an outcome in ``X`` and a weight.

    Parameters:
        X : The type of outcomes.
    """
    return frozenset[tuple[X, float]]


Subdistribution = Monad.from_maps(
    "Subdistribution",
    ob_map=dist,
    lift=lambda f: Function(
        lambda d: merge((f(x), p) for x, p in d),
        dist(untuplify(f.dom)), dist(untuplify(f.cod))),
    unit_map=lambda X: Function(
        lambda x: frozenset({(x, 1.)}), X, dist(X)),
    mult_map=lambda X: Function(
        lambda dd: merge(
            (x, p_out * p_in) for d, p_out in dd for x, p_in in d),
        dist(dist(X)), dist(X)))
"""
The subdistribution monad, sending a type ``X`` to finite subprobability
distributions over ``X``: the unit is the Dirac distribution and the
multiplication averages a distribution over distributions, allowing some
probability mass to be lost, e.g. to represent failure or divergence.
"""
