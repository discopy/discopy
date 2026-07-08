# -*- coding: utf-8 -*-

"""
The Kleisli category of a monad with tuple as tensor, i.e. the premonoidal
category of effectful Python functions with copy, discard and exponentials.

The tensor runs the effect of the left-hand channel first. The interchange law
``(f @ 1) >> (1 @ g) == (1 @ g) >> (f @ 1)`` holds if and only if the monad
is commutative, in which case the Kleisli category is monoidal rather than
merely premonoidal.

The closed structure interprets the effectful lambda calculus, i.e. the terms
of :mod:`discopy.closed`, in call-by-value order via :class:`Functor`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Channel
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        message_passing

Example
-------
>>> from discopy.kleisli.monad import Subdistribution
>>> C = Channel[Subdistribution]
>>> coin = C(lambda: {0: 0.5, 1: 0.5}, (), int)
>>> assert (coin >> C.copy(int))() == {(0, 0): 0.5, (1, 1): 0.5}
"""

from __future__ import annotations

from discopy import closed, hypergraph
from discopy.abc import ClosedCategory
from discopy.python import multiplicative
from discopy.utils import (
    MappingOrCallable, assert_isinstance, tuplify, untuplify, ar_factory)
from discopy.kleisli import channel


""" Channels have tuples of types as domain and codomain. """
Ty = tuple[type, ...]

exp = multiplicative.exp

DEFAULT_TOL, DEFAULT_MAX_ITER = 1e-12, 10 ** 6


@ar_factory
class Channel(channel.Channel, ClosedCategory):
    """
    An effectful Python function with tuple as tensor.

    Parameters:
        inside : The callable Python object inside the channel, which takes
            one value per wire in ``dom`` and returns a monadic value whose
            elements are (untuplified) tuples with one value per wire in
            ``cod``.
        dom : The domain of the channel, i.e. its tuple of input types.
        cod : The codomain of the channel, i.e. its tuple of output types.

    .. admonition:: Summary

        .. autosummary::

            tensor
            swap
            permutation
            copy
            discard
            ev
            curry
            uncurry
            fix
            trace
    """
    ob = Ty

    def tensor(self, *others: Channel) -> Channel:
        """
        The premonoidal tensor of ``n >= 1`` channels, called with ``@``.

        The effect of ``self`` runs before the effects of ``others``.

        Parameters:
            others : The other channels to compose in parallel.
        """
        if not others:
            return self
        other = others[0]
        assert_isinstance(other, Channel)
        monad = self.monad

        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            return monad.bind(self.inside(*left), lambda ys: monad.fmap(
                lambda zs: untuplify(tuplify(ys) + tuplify(zs)),
                other.inside(*right)))
        result = type(self)(inside, self.dom + other.dom,
                            self.cod + other.cod)
        return result.tensor(*others[1:])

    @classmethod
    def permutation(cls, xs: list[int], dom: Ty) -> Channel:
        """
        The pure channel that encodes a given permutation, with the same
        convention as :meth:`discopy.symmetric.Diagram.permutation`,
        i.e. ``output[i] == input[xs[i]]``.

        Parameters:
            xs : A list of integers representing a permutation.
            dom : A tuple of types of the same length as ``xs``.
        """
        dom = tuplify(dom)
        if sorted(xs) != list(range(len(dom))):
            raise ValueError(
                f"Expected a permutation of range({len(dom)}), got {xs}.")
        cod = tuple(dom[i] for i in xs)
        result = cls(lambda *vals: cls.monad.pure(
            untuplify(tuple(vals[i] for i in xs))), dom, cod)
        result.is_permutation_of = (tuple(xs), dom)
        return result

    def dagger(self) -> Channel:
        """ The inverse of a permutation channel. """
        if getattr(self, "is_permutation_of", None) is None:
            raise ValueError(f"{self} is not a permutation.")
        xs, _ = self.is_permutation_of
        return type(self).permutation(
            [xs.index(i) for i in range(len(xs))], self.cod)

    @classmethod
    def swap(cls, x: Ty, y: Ty) -> Channel:
        """
        The pure channel that swaps two tuples of types ``x`` and ``y``.

        Parameters:
            x : The tuple of types on the left.
            y : The tuple of types on the right.
        """
        x, y = map(tuplify, (x, y))
        return cls.permutation(
            list(range(len(x), len(x) + len(y))) + list(range(len(x))), x + y)

    braid = swap

    @classmethod
    def copy(cls, x: Ty, n=2) -> Channel:
        """
        The pure channel making ``n`` copies of a tuple of types ``x``.

        Parameters:
            x : The tuple of types to copy.
            n : The number of copies.
        """
        x = tuplify(x)
        return cls(lambda *xs: cls.monad.pure(untuplify(n * xs)), x, n * x)

    @classmethod
    def discard(cls, dom: Ty) -> Channel:
        """
        The pure channel discarding a tuple of types, i.e. zero copies.

        Parameters:
            dom : The tuple of types to discard.
        """
        return cls.copy(dom, 0)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left=True) -> Channel:
        """
        The evaluation channel, i.e. apply an effectful closure to arguments.

        Parameters:
            base : The output type.
            exponent : The input type.
            left : Whether to take the closure on the left or right.
        """
        base, exponent = map(tuplify, (base, exponent))
        if left:
            dom = exp(base, exponent) + exponent
            return cls(lambda f, *xs: f(*xs), dom, base)
        dom = exponent + exp(base, exponent)
        return cls(lambda *xs: xs[-1](*xs[:-1]), dom, base)

    def curry(self, n=1, left=True) -> Channel:
        """
        Currying, i.e. turn an effectful function into a pure channel that
        outputs an effectful closure.

        Parameters:
            n : The number of types to curry.
            left : Whether to curry on the left or right.
        """
        monad = self.monad
        if left:
            dom = self.dom[:len(self.dom) - n]
            cod = exp(self.cod, self.dom[len(self.dom) - n:])
        else:
            dom, cod = self.dom[n:], exp(self.cod, self.dom[:n])
        return type(self)(dom=dom, cod=cod, inside=lambda *xs: monad.pure(
            lambda *ys: self.inside(*(xs + ys) if left else (ys + xs))))

    def uncurry(self, left=True) -> Channel:
        """
        Uncurrying, i.e. turn a closure-valued channel into an effectful
        function of its arguments.

        Parameters:
            left : Whether to uncurry on the left or right.
        """
        inside = self.cod[0].__args__
        base, exponent = inside[-1].__args__, inside[:-1]
        return self @ exponent >> type(self).ev(base, exponent) if left\
            else exponent @ self >> type(self).ev(base, exponent, left=False)

    def fix(self, n=1, tol=DEFAULT_TOL, max_iter=DEFAULT_MAX_ITER) -> Channel:
        """
        The parameterised fixed point of a channel, by monadic Kleene
        iteration: ``m = inside(*xs)`` then
        ``m = bind(m, y -> inside(*xs, y))`` until the monadic value is
        stable up to :meth:`Monad.allclose`.

        Parameters:
            n : The number of types to take the fixed point over.
            tol : The tolerance up to which to compare iterations.
            max_iter : The maximum number of iterations before raising
                ``RuntimeError``.
        """
        if n == 0:
            return self
        monad = self.monad

        def inside(*xs):
            result = self.inside(*xs)
            for _ in range(max_iter):
                update = monad.bind(
                    result, lambda y: self.inside(*xs + (y, )))
                if monad.allclose(update, result, tol):
                    return update
                result = update
            raise RuntimeError("Fixed point did not converge.")
        return type(self)(inside, self.dom[:-1], self.cod).fix(
            n - 1, tol, max_iter)

    def trace(self, n=1, left=False,
              tol=DEFAULT_TOL, max_iter=DEFAULT_MAX_ITER) -> Channel:
        """
        The multiplicative trace of a channel, computed with :meth:`fix`.

        Parameters:
            n : The number of types to trace over.
            left : Whether to trace on the left or right, only ``right`` is
                implemented.
            tol : The tolerance up to which to compare iterations.
            max_iter : The maximum number of iterations.
        """
        if left:
            raise NotImplementedError
        dom, cod, traced = self.dom[:-n], self.cod[:-n], self.dom[-n:]
        fixed = (self >> self.discard(cod) @ traced).fix(
            tol=tol, max_iter=max_iter)
        return self.copy(dom) >> dom @ fixed\
            >> self >> cod @ self.discard(traced)

    exp = over = under = staticmethod(exp)


class Functor(closed.Functor):
    """
    A functor from :class:`discopy.closed.Diagram` into the Kleisli category
    of a monad, i.e. the call-by-value semantics of effectful lambda terms.

    Parameters:
        ob (Mapping[closed.Ty, Ty]) :
            Map from atomic :class:`discopy.closed.Ty` to tuples of types.
        ar : Map from :class:`discopy.closed.Box` to channels; raw monadic
            callables are also accepted and coerced into channels.
        cod : The codomain, some ``Channel[M]``.

    Example
    -------
    >>> from discopy.kleisli.monad import Subdistribution
    >>> X = closed.Ty('X')
    >>> coin = closed.Constant('coin', X)
    >>> add = closed.Constant('add', X >> (X >> X))
    >>> M = Subdistribution
    >>> F = Functor({X: int}, {
    ...     coin: lambda: {0: 0.5, 1: 0.5},
    ...     add: lambda: M.pure(lambda a: M.pure(lambda b: M.pure(a + b)))},
    ...     cod=Channel[M])
    >>> term = X(lambda v: add(v)(v))(coin)
    >>> assert F(term)() == {0: 0.5, 2: 0.5}
    """
    dom = closed.Diagram

    def __init__(self, ob=None, ar=None, dom=None, cod=None):
        super().__init__(ob, ar, dom=dom, cod=cod)
        raw = self.ar_map
        self.ar_map = MappingOrCallable(lambda box: self.__coerce__(raw, box))

    def __coerce__(self, raw, box):
        result = raw[box]
        return result if isinstance(result, self.cod)\
            else self.cod(result, self(box.dom), self(box.cod))


def message_passing(
        functor: Functor, diagram,
        tol=DEFAULT_TOL, max_iter=DEFAULT_MAX_ITER) -> Channel:
    """
    Evaluate a diagram (or hypergraph) by monadic message passing over its
    hypergraph, i.e. fire each box in diagram order as soon as its input
    spiders carry values, keeping the joint monadic state over an append-only
    environment of spider values.

    Feedback spiders, i.e. those consumed before they are produced, come from
    traces; the first round fires their consumer boxes with trailing arguments
    missing (the same defaults convention as :meth:`Channel.fix`), later
    rounds re-inject the previous round's per-spider messages and iterate
    until they are stable up to :meth:`Monad.allclose`.

    The result agrees with recursive evaluation through ``functor`` for the
    given diagram order; two interchanged diagrams have equal hypergraphs, so
    the hypergraph semantics is well-defined if and only if the monad is
    commutative.

    Parameters:
        functor : The functor mapping boxes to channels.
        diagram : The diagram, or its hypergraph, to evaluate.
        tol : The tolerance up to which to compare feedback messages.
        max_iter : The maximum number of rounds.
    """
    graph = diagram if isinstance(diagram, hypergraph.Hypergraph)\
        else diagram.to_hypergraph()
    cls, monad = functor.cod, functor.cod.monad
    channels = [functor(box) for box in graph.boxes]
    dom, cod = functor(graph.dom), functor(graph.cod)
    dom_wires, box_wires, cod_wires = graph.wires

    produced_at = {spider: -1 for spider in dom_wires}
    for depth, (_, outs) in enumerate(box_wires):
        for spider in outs:
            produced_at.setdefault(spider, depth)
    feedback = []
    for depth, (ins, _) in enumerate(box_wires):
        for spider in ins:
            if spider not in produced_at:
                raise ValueError(f"Spider {spider} has no producer.")
            if produced_at[spider] >= depth and spider not in feedback:
                feedback.append(spider)
    for spider in cod_wires:
        if spider not in produced_at:
            raise ValueError(f"Spider {spider} has no producer.")

    def run_round(xs, messages):
        slots, m = {}, monad.pure(tuple(xs))
        for i, spider in enumerate(dom_wires):
            slots.setdefault(spider, i)
        length = len(xs)
        for spider in feedback if messages else ():
            m = monad.bind(m, lambda env, msg=messages[spider]:
                           monad.fmap(lambda v, env=env: env + (v, ), msg))
            slots[spider] = length
            length += 1
        for box, (ins, outs) in zip(channels, box_wires):
            present = [spider for spider in ins if spider in slots]
            if list(ins[:len(present)]) != present:
                raise ValueError(
                    "Missing input spiders must be a trailing suffix, "
                    f"got {ins} with {present} present.")
            arg_slots = tuple(slots[spider] for spider in present)

            def fire(env, box=box, arg_slots=arg_slots):
                return box.inside(*[env[i] for i in arg_slots])
            m = monad.bind(m, lambda env, fire=fire: monad.fmap(
                lambda ys, env=env: env + tuplify(ys), fire(env)))
            for j, spider in enumerate(outs):
                slots[spider] = length + j
            length += len(outs)
        return m, slots

    def inside(*xs):
        messages = {}
        for _ in range(max_iter):
            m, slots = run_round(xs, messages)
            if not feedback:
                break
            updated = {
                spider: monad.fmap(lambda env, i=slots[spider]: env[i], m)
                for spider in feedback}
            if messages and all(
                    monad.allclose(updated[spider], messages[spider], tol)
                    for spider in feedback):
                break
            messages = updated
        else:
            raise RuntimeError("Message passing did not converge.")
        return monad.fmap(lambda env: untuplify(
            tuple(env[slots[spider]] for spider in cod_wires)), m)

    return cls(inside, dom, cod)
