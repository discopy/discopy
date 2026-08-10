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

A monad may also come with an :attr:`Monad.iterate` operator, i.e. the
extra structure of an `Elgot monad
<https://ncatlab.org/nlab/show/Elgot+monad>`_ needed to trace the Kleisli
category, see :mod:`discopy.kleisli.additive`.

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
    Seed
    make_monad
    make_state
    iterate_maybe
    iterate_powerset
    iterate_subdistribution
"""
from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from discopy.cat import Transformation
from discopy.python.function import EndoFunctor, Function
from discopy.utils import (
    assert_isinstance, factory_name, tuplify, untuplify)


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
        iterate : The optional iteration operator, see :attr:`iterate`.

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
    iterate: Callable = None
    """
    The iteration operator, i.e. the extra structure of an `Elgot monad
    <https://ncatlab.org/nlab/show/Elgot+monad>`_, or ``None`` when the
    monad does not come with one.

    It takes a loop body ``step : Z -> M(Z)`` together with a predicate
    ``exits`` telling whether an outcome has left the loop, and returns the
    function sending a monadic value to the one where every looping outcome
    has been fed back into ``step`` until it exits. Writing ``Z = Y + U``
    with ``exits`` the characteristic function of ``Y``, this is the
    iteration operator

    .. math::
        (-)^\\dagger : (U \\to M(Y + U)) \\to (U \\to M(Y))

    which does not follow from ``(functor, unit, mult)``: a monadic value
    holds both exiting and looping outcomes at once and the monoid
    structure alone cannot tell them apart. This is what makes
    :meth:`discopy.kleisli.additive.Channel.trace` defined exactly for the
    monads that supply one, see :func:`iterate_maybe`,
    :func:`iterate_powerset` and :func:`iterate_subdistribution`.
    """

    def __init__(
            self, name: str, functor: EndoFunctor,
            unit: Transformation, mult: Transformation,
            iterate: Callable = None):
        assert_isinstance(functor, EndoFunctor)
        assert_isinstance(unit, Transformation)
        assert_isinstance(mult, Transformation)
        self.__name__ = self.name = name
        self.functor, self.unit, self.mult = functor, unit, mult
        self.iterate = iterate

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
            cls, name, ob_map, lift, unit_map, mult_map,
            iterate: Callable = None) -> Monad:
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
            iterate : The optional iteration operator, see
                :attr:`Monad.iterate`.
        """
        unwrap = lambda X: untuplify(tuplify(X))
        functor = EndoFunctor(lambda X: (ob_map(untuplify(X)), ), lift)
        unit = Transformation(
            lambda X: unit_map(unwrap(X)), EndoFunctor.id(), functor)
        mult = Transformation(
            lambda X: mult_map(unwrap(X)), functor.then(functor), functor)
        return cls(name, functor, unit, mult, iterate)

    def __repr__(self):
        return f"Monad({self.name!r})"

    def __str__(self):
        return self.name


def make_monad(
        name, ob_map, lift, unit_map, mult_map,
        iterate: Callable = None) -> Monad:
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
        iterate : The optional iteration operator, see
            :attr:`Monad.iterate`.
    """
    return Monad.from_maps(
        name, ob_map, lift, unit_map, mult_map, iterate)


def iterate_maybe(step: Callable, exits: Callable) -> Callable:
    """
    The iteration operator of the :attr:`Maybe` monad, i.e. a ``while``
    loop: there is at most one outcome, so it either exits, fails or
    diverges.

    Parameters:
        step : The loop body, from an outcome to a maybe-outcome.
        exits : The predicate telling whether an outcome has left the loop.

    Example
    -------
    >>> step, exits = lambda x: x + 1, lambda x: x >= 3
    >>> assert iterate_maybe(step, exits)(0) == 3
    >>> assert iterate_maybe(step, exits)(4) == 4
    >>> assert iterate_maybe(lambda x: None, exits)(0) is None
    """
    def inside(m):
        while m is not None and not exits(m):
            m = step(m)
        return m
    return inside


def iterate_powerset(step: Callable, exits: Callable) -> Callable:
    """
    The iteration operator of the :attr:`Powerset` monad, i.e. the set of
    outcomes reachable in finitely many steps, computed breadth-first.

    An outcome that has already been stepped is never stepped again, so a
    loop that cycles back to a state it has visited contributes nothing
    rather than diverging, i.e. this is the least fixed point.

    Parameters:
        step : The loop body, from an outcome to a set of outcomes.
        exits : The predicate telling whether an outcome has left the loop.

    Example
    -------
    A walk on the integers that exits as soon as it is out of ``range(3)``:

    >>> step = lambda x: frozenset({x - 1, x + 1})
    >>> exits = lambda x: x not in range(3)
    >>> assert iterate_powerset(step, exits)(
    ...     frozenset({1})) == frozenset({-1, 3})

    A cycle never exits, so it contributes nothing:

    >>> assert iterate_powerset(lambda x: frozenset({1 - x}), exits)(
    ...     frozenset({0})) == frozenset()
    """
    def inside(m):
        done = frozenset(x for x in m if exits(x))
        pending = seen = frozenset(m) - done
        while pending:
            outcomes = frozenset().union(*map(step, pending))
            done |= frozenset(x for x in outcomes if exits(x))
            pending = frozenset(
                x for x in outcomes if not exits(x)) - seen
            seen |= pending
        return done
    return inside


Maybe = Monad.from_maps(
    "Maybe",
    ob_map=lambda X: X | None,
    lift=lambda f: Function(
        lambda x: None if x is None else f(x),
        untuplify(f.dom) | None, untuplify(f.cod) | None),
    unit_map=lambda X: Function(lambda x: x, X, X | None),
    mult_map=lambda X: Function(lambda x: x, (X | None) | None, X | None),
    iterate=iterate_maybe)
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
        frozenset[frozenset[X]], frozenset[X]),
    iterate=iterate_powerset)
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


def iterate_subdistribution(
        step: Callable, exits: Callable,
        tolerance: float = 1e-12) -> Callable:
    """
    The iteration operator of the :attr:`Subdistribution` monad, i.e. the
    execution formula: push the mass that is still looping through ``step``
    and accumulate the mass that exits, until what is left is below
    ``tolerance``.

    The looping mass goes to zero, and the iteration stops, exactly when
    the loop halts almost surely or loses its mass: this is what it means
    for the monad to be sub-additive, and the result is then within
    ``tolerance`` of the limit. A loop that goes around forever with
    positive probability keeps its mass and the iteration does not
    terminate, just like the ``while`` loop it generalises.

    Parameters:
        step : The loop body, from an outcome to a subdistribution.
        exits : The predicate telling whether an outcome has left the loop.
        tolerance : The mass below which the iteration stops.

    Example
    -------
    A fair coin flipped until it comes up heads, giving up after two flips:

    >>> flip = lambda x: frozenset({
    ...     (("heads", x[1]), .5), (("tails", x[1] + 1), .5)})
    >>> exits = lambda x: x[0] == "heads" or x[1] > 1
    >>> assert iterate_subdistribution(flip, exits)(
    ...     frozenset({(("tails", 0), 1.)})) == frozenset({
    ...         (("heads", 0), .5), (("heads", 1), .25), (("tails", 2), .25)})

    A loop that never exits keeps none of the mass it loses:

    >>> assert iterate_subdistribution(
    ...     lambda x: frozenset({(x + 1, .5)}), lambda x: False)(
    ...         frozenset({(0, 1.)})) == frozenset()
    """
    def inside(m):
        done = merge((x, p) for x, p in m if exits(x))
        pending = frozenset((x, p) for x, p in m if not exits(x))
        while sum(p for _, p in pending) > tolerance:
            outcomes = merge(
                (y, p * q) for x, p in pending for y, q in step(x))
            done = merge(tuple(done) + tuple(
                (y, q) for y, q in outcomes if exits(y)))
            pending = frozenset(
                (y, q) for y, q in outcomes if not exits(y))
        return done
    return inside


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
        dist(dist(X)), dist(X)),
    iterate=iterate_subdistribution)
"""
The subdistribution monad, sending a type ``X`` to finite subprobability
distributions over ``X``: the unit is the Dirac distribution and the
multiplication averages a distribution over distributions, allowing some
probability mass to be lost, e.g. to represent failure or divergence.
"""


def state_map(f: Callable, m: Callable) -> Callable:
    """
    The functorial action of a state monad, i.e. apply ``f`` to the value
    returned by the computation ``m`` and thread the state through.

    Parameters:
        f : The function to apply to the value.
        m : The computation from a state to a value and the next state.
    """
    def inside(state):
        value, next_state = m(state)
        return f(value), next_state
    return inside


def state_join(mm: Callable) -> Callable:
    """
    The multiplication of a state monad, i.e. run the outer computation
    ``mm`` then the inner one it returns on the state it left.

    Parameters:
        mm : The computation returning a computation.
    """
    def inside(state):
        m, next_state = mm(state)
        return m(next_state)
    return inside


def make_state(S: type) -> Monad:
    """
    The state monad for a type ``S`` of states, sending a type ``X`` to the
    type ``S -> (X, S)`` of computations reading a state and returning a
    value together with the next state.

    Parameters:
        S : The type of states.

    Note
    ----
    The state monad is not commutative: two channels tensored in its Kleisli
    category run their effects in a given order, so the two biased tensors
    differ, see :mod:`discopy.kleisli.multiplicative`. It comes with no
    :attr:`Monad.iterate`, so tracing one of its channels raises rather
    than guessing at the semantics, see
    :meth:`discopy.kleisli.additive.Channel.trace`.

    Example
    -------
    >>> State = make_state(int)
    >>> assert State(str) == (Callable[[int], tuple[str, int]], )

    The unit leaves the state untouched and the multiplication runs the
    outer computation before the inner one:

    >>> tick = lambda x: lambda s: (x, s + 1)
    >>> assert State.unit(str)("egg")(0) == ("egg", 0)
    >>> assert State.mult(str)(lambda s: (tick("egg"), s + 1))(0) == ("egg", 2)
    """
    ob_map = lambda X: Callable[[S], tuple[X, S]]
    return Monad.from_maps(
        f"State[{factory_name(S)}]",
        ob_map=ob_map,
        lift=lambda f: Function(
            lambda m: state_map(f, m),
            ob_map(untuplify(f.dom)), ob_map(untuplify(f.cod))),
        unit_map=lambda X: Function(
            lambda x: lambda state: (x, state), X, ob_map(X)),
        mult_map=lambda X: Function(
            state_join, ob_map(ob_map(X)), ob_map(X)))


Seed = make_state(int)
"""
The state monad on integer seeds, i.e. the monad of seeded randomness: a
random computation is the pure function sending a seed to a sample and the
seed for the next sample, see :func:`uniform` and :func:`sample`.
"""


def uniform(seed: int) -> tuple[float, int]:
    """
    A sample from the uniform distribution on the unit interval together
    with the seed for the next sample, i.e. a computation in the
    :attr:`Seed` monad.

    Parameters:
        seed : The seed for this sample.

    Example
    -------
    >>> assert uniform(420) == uniform(420) != uniform(1337)
    >>> assert 0 <= uniform(420)[0] < 1
    """
    generator = random.Random(seed)
    return generator.random(), generator.getrandbits(64)


def sample(d: frozenset) -> Callable:
    """
    Draw an outcome from a subdistribution ``d`` by `inverse transform
    sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_,
    i.e. a computation in the :attr:`Seed` monad.

    The outcomes are ordered by their representation so that the sample
    depends on the seed only, never on the iteration order of a
    :class:`frozenset`. The missing mass of a subdistribution is drawn as
    ``None``, i.e. sampling lands in :attr:`Maybe` composed with
    :attr:`Seed`.

    Parameters:
        d : The subdistribution to sample from, i.e. a finite set of pairs
            of an outcome and its weight.

    Example
    -------
    >>> coin = frozenset({("heads", .5), ("tails", .5)})
    >>> assert sample(coin)(420)[0] in ("heads", "tails")
    >>> assert sample(frozenset({("heads", .5)}))(420)[0] in ("heads", None)
    """
    outcomes = sorted(d, key=lambda pair: repr(pair[0]))

    def inside(seed):
        draw, next_seed = uniform(seed)
        for outcome, weight in outcomes:
            draw -= weight
            if draw < 0:
                return outcome, next_seed
        return None, next_seed
    return inside
