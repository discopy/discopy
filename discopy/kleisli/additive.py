# -*- coding: utf-8 -*-

"""
The Kleisli category of a monad with disjoint union as tensor, i.e. effectful
Python functions between tagged unions, with a trace given by iteration.

The trace, i.e. a while loop, requires the monad to have the sub-additive
structure ``zero, plus, mass, support``: at each step the mass that exits the
loop is accumulated with ``plus`` and the loop stops when the mass still
alive is at most ``tol``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Channel

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        token_passing

Example
-------
>>> from discopy.kleisli.monad import Maybe, Nothing
>>> C = Channel[Maybe]
>>> count = C(lambda n, tag: (n, 0) if n <= 0 else (n - 1, 1),
...           (int, int), (int, int))
>>> assert count.trace()(3) == (0, 0) and count.trace()(0) == (0, 0)
"""

from __future__ import annotations

from discopy import hypergraph
from discopy.abc import SymmetricCategory
from discopy.utils import assert_isinstance, tuplify, ar_factory
from discopy.kleisli import channel


""" Tuples of types interpreted as disjoint union. """
Ty = tuple[type, ...]

DEFAULT_TOL, DEFAULT_MAX_ITER = 1e-12, 10 ** 6


@ar_factory
class Channel(channel.Channel, SymmetricCategory):
    """
    An effectful Python function with disjoint union as tensor.

    Parameters:
        inside : The callable Python object inside the channel, which takes a
            pair of a value and a tag indexing ``dom`` and returns a monadic
            value whose elements are pairs ``(value, tag)`` indexing ``cod``.
        dom : The domain of the channel, i.e. its tuple of input types.
        cod : The codomain of the channel, i.e. its tuple of output types.

    .. admonition:: Summary

        .. autosummary::

            tensor
            swap
            merge
            inject
            trace
    """
    ob = Ty

    def __call__(self, obj, tag=0):
        return self.inside(obj, tag)

    def tensor(self, *others: Channel) -> Channel:
        """
        The disjoint union of ``n >= 1`` channels, called with ``@``.

        Parameters:
            others : The other channels to compose in parallel.
        """
        if not others:
            return self
        other = others[0]
        assert_isinstance(other, Channel)
        monad = self.monad

        def inside(obj, tag):
            if tag < len(self.dom):
                return self.inside(obj, tag)
            return monad.fmap(
                lambda pair: (pair[0], pair[1] + len(self.cod)),
                other.inside(obj, tag - len(self.dom)))
        result = type(self)(inside, self.dom + other.dom,
                            self.cod + other.cod)
        return result.tensor(*others[1:])

    @classmethod
    def swap(cls, x: Ty, y: Ty) -> Channel:
        """
        The pure channel retagging a disjoint union from ``x + y`` to
        ``y + x``.

        Parameters:
            x : The tuple of types on the left.
            y : The tuple of types on the right.
        """
        x, y = map(tuplify, (x, y))

        def inside(obj, tag):
            return cls.monad.pure(
                (obj, tag + len(y) if tag < len(x) else tag - len(x)))
        result = cls(inside, x + y, y + x)
        result.is_swap_of = (x, y)
        return result

    braid = swap

    def dagger(self) -> Channel:
        """ The inverse of a swap channel. """
        if getattr(self, "is_swap_of", None) is None:
            raise ValueError(f"{self} is not a swap.")
        return type(self).swap(*self.is_swap_of[::-1])

    @classmethod
    def merge(cls, x: Ty, n=2) -> Channel:
        """
        The pure channel merging ``n`` copies of a tuple of types ``x``.

        Parameters:
            x : The tuple of types to merge.
            n : The number of copies.
        """
        x = tuplify(x)
        return cls(
            lambda obj, tag: cls.monad.pure((obj, tag % len(x))), n * x, x)

    @classmethod
    def inject(cls, ty: Ty, i: int) -> Channel:
        """
        The pure channel injecting the ``i``-th component into a disjoint
        union ``ty``.

        Parameters:
            ty : The tuple of types to inject into.
            i : The index of the component.
        """
        ty = tuplify(ty)
        return cls(lambda obj, tag=0: cls.monad.pure((obj, i)), ty[i], ty)

    def trace(self, n=1, left=False,
              tol=DEFAULT_TOL, max_iter=DEFAULT_MAX_ITER) -> Channel:
        """
        The additive trace of a channel, i.e. a while loop: outputs with a
        tag in the last ``n`` components are fed back as inputs to the last
        ``n`` components until the monadic mass still alive is at most
        ``tol``.

        Idempotent monads with ``filter_repeats`` drop already-visited
        elements, so deterministic cycles converge (e.g. to ``Nothing`` for
        the maybe monad); the subdistribution monad converges by mass decay.

        Parameters:
            n : The number of types to trace over.
            left : Whether to trace on the left or right, only ``right`` is
                implemented.
            tol : The mass below which the loop terminates.
            max_iter : The maximum number of iterations before raising
                ``RuntimeError``.
        """
        if left:
            raise NotImplementedError
        monad = self.monad
        if not monad.is_additive:
            raise ValueError(
                f"The additive trace requires a sub-additive monad, "
                f"got {monad}.")
        dom, cod = self.dom[:-n], self.cod[:-n]

        def inside(obj, tag):
            result, m, seen = monad.zero(), self.inside(obj, tag), set()
            for _ in range(max_iter):
                exited = monad.bind(m, lambda pair: monad.pure(pair)
                                    if pair[1] < len(cod) else monad.zero())
                result = monad.plus(result, exited)
                live = monad.bind(m, lambda pair: monad.zero()
                                  if pair[1] < len(cod) else monad.pure(pair))
                if monad.filter_repeats:
                    live = monad.bind(
                        live, lambda pair: monad.zero()
                        if pair in seen else monad.pure(pair))
                    seen.update(monad.support(live))
                if monad.mass(live) <= tol:
                    return result
                m = monad.bind(live, lambda pair: self.inside(
                    pair[0], len(self.dom) - n + (pair[1] - len(cod))))
            raise RuntimeError("Additive trace did not converge.")
        return type(self)(inside, dom, cod)


def token_passing(functor, diagram,
                  tol=DEFAULT_TOL, max_iter=DEFAULT_MAX_ITER) -> Channel:
    """
    Evaluate a diagram (or hypergraph) by monadic token passing over its
    hypergraph: the monadic state ranges over pairs ``(spider, value)`` of a
    position and a value, tokens step through the unique consumer of their
    spider, and tokens on output spiders accumulate into the result.

    Cycles in the hypergraph, i.e. traces, need no special casing: the live
    monadic value updates until it converges to zero with the same
    ``zero, plus, mass, filter_repeats`` machinery as :meth:`Channel.trace`.

    Parameters:
        functor : The functor mapping boxes to channels, e.g. a
            :class:`discopy.symmetric.Functor` with some ``Channel[M]`` as
            codomain.
        diagram : The diagram, or its hypergraph, to evaluate.
        tol : The mass below which the loop terminates.
        max_iter : The maximum number of steps.
    """
    graph = diagram if isinstance(diagram, hypergraph.Hypergraph)\
        else diagram.to_hypergraph()
    cls, monad = functor.cod, functor.cod.monad
    channels = [functor(box) for box in graph.boxes]
    dom, cod = functor(graph.dom), functor(graph.cod)
    dom_wires, box_wires, cod_wires = graph.wires

    if len(set(cod_wires)) != len(cod_wires):
        raise ValueError("Copied outputs are not additive.")
    out_index = {spider: k for k, spider in enumerate(cod_wires)}
    consumer = {}
    for depth, (ins, _) in enumerate(box_wires):
        for i, spider in enumerate(ins):
            if spider in consumer or spider in out_index:
                raise ValueError(
                    f"Spider {spider} has more than one consumer, "
                    "the diagram is not additive.")
            consumer[spider] = (depth, i)

    def step(token):
        spider, value = token
        if spider not in consumer:
            raise ValueError(f"Token stuck on spider {spider}.")
        depth, i = consumer[spider]
        outs = box_wires[depth][1]
        return monad.fmap(
            lambda pair: (outs[pair[1]], pair[0]),
            channels[depth].inside(value, i))

    def inside(obj, tag):
        result, seen = monad.zero(), set()
        m = monad.pure((dom_wires[tag], obj))
        for _ in range(max_iter):
            exited = monad.bind(
                m, lambda token: monad.pure((token[1], out_index[token[0]]))
                if token[0] in out_index else monad.zero())
            result = monad.plus(result, exited)
            live = monad.bind(m, lambda token: monad.zero()
                              if token[0] in out_index else monad.pure(token))
            if monad.filter_repeats:
                live = monad.bind(live, lambda token: monad.zero()
                                  if token in seen else monad.pure(token))
                seen.update(monad.support(live))
            if monad.mass(live) <= tol:
                return result
            m = monad.bind(live, step)
        raise RuntimeError("Token passing did not converge.")
    return cls(inside, dom, cod)
