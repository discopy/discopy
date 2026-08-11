# -*- coding: utf-8 -*-

"""
What a whole map does in one round: the global transition.

:mod:`discopy.neural.parametric` says what a *box* is -- an interaction
:math:`\\Phi_f` on its boundary.  A map wires the boundaries together, and
one synchronous round of message passing is the two halves in sequence:
every box interacts with the messages on its own ports, then the wires
carry each emission to the other end.  Writing :math:`\\Phi_\\theta` for the
parallel application of every local interaction and :math:`\\sigma_D` for
the permutation of the flat state induced by the wiring,

.. math:: T_{D,\\theta} = \\sigma_D \\circ \\Phi_\\theta : S \\to S

on the state object :math:`S = \\bigoplus_p \\mathbb{R}^{w_p}`, one summand
per port.  This is the execution formula of the geometry of interaction,
see :cite:t:`Abramsky96`.

When an initial message vector :math:`i` is re-injected -- ``inject=True``
in :meth:`~discopy.neural.CMap.forward` -- the round is

.. math:: T_{D,\\theta,i}(s) = \\sigma_D(\\Phi_\\theta(s)) + i,

an affine, not a linear, dependence on :math:`i`: the vector is added back
to the *whole* state after routing, every round.  On an open map the
boundary input ports are not owned by a box and re-emit the input ``x``
instead, so the round reads
:math:`T(s) = \\sigma_D(\\Phi_\\theta(s) \\uplus x) + i`.

Finite iteration, not a fixed point
-----------------------------------

Running ``n`` rounds computes the finite iterate :math:`T^n(s_0)` --
:class:`Iteration` -- and nothing more.  No fixed point is solved, no
contraction is assumed, and nothing here converges by fiat.  What *is*
free is resumption, :math:`T^{a+b} = T^b \\circ T^a`, which is why a
segmented outer loop can stop after ``a`` rounds and carry on, and it is
the law :meth:`Iteration.__rshift__` records.

Resumption is a law of *one* transition, and reinjection is part of the
transition: :math:`T_{D,\\theta,i}` and :math:`T_{D,\\theta,i'}` are two
different maps.  A run resumed from its own carried state therefore only
resumes when ``inject`` is off -- which is why the schedules that resume
are the ones that pass ``inject=False`` and carry their inputs on a state
channel instead, while a schedule with ``inject=True`` runs exactly one
:meth:`~discopy.neural.engine.Engine.advance` per forward pass.

Three notions that are easy to conflate, kept apart:

* a **categorical trace** is a structural operation on wiring.  A
  self-wired pair of ports *is* the trace of the compact target, and a
  functor into :mod:`discopy.neural` preserves it strictly for free --
  :mod:`discopy.neural.skeleton` checks that equation as a doctest.
* a **persistent state channel** is what that same pair does across
  rounds: what a box writes on one end it reads on the other one round
  later.  Delayed feedback, in the sense of
  :mod:`discopy.feedback`, not an equation to solve.
* a **fixed point** of :math:`T` is a third thing, and nothing in this
  package computes one.  If some :math:`T` happens to be a contraction
  then :math:`T^n` converges, but that is an analytic property of the
  learned weights to be measured, never something the category supplies.

Nothing here computes either.  :meth:`~discopy.neural.CMap.forward` is the
one implementation of :math:`T`; :meth:`~discopy.neural.engine.Engine.
advance` is the engine's adapter for :math:`T^n`.  This module reads their
metadata: :func:`from_map` builds the specification of an interpreted map,
and never touches a tensor.

Like :mod:`discopy.neural.core`, this module does not import ``torch``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Transition
    Iteration

Example
-------

>>> from discopy.neural import CMap, Dim, Network
>>> cell = Network("cell", Dim(0), Dim(2) ** 2)
>>> ring = CMap.from_wiring(
...     (cell, cell), [((0, 0), (1, 1)), ((0, 1), (1, 0))])
>>> round_ = from_map(ring)
>>> round_.state, round_.width, round_.is_closed
(Dim(2, 2, 2, 2), 8, True)
>>> round_.routing
(3, 2, 1, 0)
>>> round_.iterate(6) == round_.iterate(2) >> round_.iterate(4)
True
"""

from __future__ import annotations

from dataclasses import dataclass

from discopy.cmap import PortKind
from discopy.neural.core import Dim
from discopy.neural.parametric import interaction_spec


@dataclass(frozen=True)
class Transition:
    """
    One round of a map, as a specification: :math:`T_\\theta = \\sigma \\circ
    \\Phi_\\theta` on the state object of its ports.

    Parameters:
        local : The interaction of each box, in box order.
        routing : The wiring :math:`\\sigma`, as a fixpoint-free involution
                  on ports.
        widths : The width each port carries.
        inputs : The boundary input ports, empty on a closed map.
        outputs : The boundary output ports, empty on a closed map.
        inject : Whether the round re-adds the initial messages, i.e.
                 whether the transition is :math:`\\sigma(\\Phi(s)) + i`
                 rather than :math:`\\sigma(\\Phi(s))`.

    Note
    ----
    There is deliberately no parameter object here.  Sites sharing a module
    share their parameters -- that is what makes the model size independent
    of the map size -- so :math:`\\theta` is *not* the product of the
    parameter objects of :attr:`local`, and
    :meth:`~discopy.neural.CMap.parameters` is the only authority on what
    it is.

    Example
    -------
    >>> from discopy.neural import CMap, Dim, Network
    >>> f = Network("f", Dim(2), Dim(3))
    >>> open_map = from_map(f.to_map())
    >>> open_map.widths, open_map.is_closed
    ((2, 2, 3, 3), False)
    >>> open_map.inputs, open_map.outputs
    ((0,), (3,))
    >>> [spec.boundary for spec in open_map.local]
    [Dim(2, 3)]
    """

    local: tuple
    routing: tuple[int, ...]
    widths: tuple[int, ...]
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    inject: bool = True

    @property
    def state(self) -> Dim:
        """ The state object :math:`S`, one summand per port. """
        return Dim(*self.widths)

    @property
    def width(self) -> int:
        """ The flat width of the state, i.e. of one message vector. """
        return sum(self.widths)

    @property
    def n_ports(self) -> int:
        """ The number of ports the state has a summand for. """
        return len(self.widths)

    @property
    def is_closed(self) -> bool:
        """
        Whether the map has no boundary, so that every port is owned by a
        box and the round is a transition of the state alone.
        """
        return not (self.inputs or self.outputs)

    def is_involution(self) -> bool:
        """
        Whether :attr:`routing` is a fixpoint-free involution, i.e. whether
        every port is wired to exactly one other and never to itself.

        Example
        -------
        >>> Transition((), (1, 0), (2, 2)).is_involution()
        True
        >>> Transition((), (0, 1), (2, 2)).is_involution()
        False
        """
        return all(self.routing[other] == port and other != port
                   for port, other in enumerate(self.routing))

    def iterate(self, rounds: int) -> Iteration:
        """
        The finite iterate :math:`T^{\\text{rounds}}`.

        Parameters:
            rounds : The number of rounds, non-negative.
        """
        return Iteration(self, rounds)


@dataclass(frozen=True)
class Iteration:
    """
    A finite number of rounds of one transition, :math:`T^n`.

    This is a denotation, not a limit: :math:`T^n(s_0)` is what ``n``
    rounds compute, whether or not the sequence goes anywhere.  What holds
    unconditionally is resumption, :math:`T^{a+b} = T^b \\circ T^a`, spelled
    ``>>``.

    Parameters:
        transition : The round to iterate.
        rounds : How many times, non-negative.

    Example
    -------
    >>> loop = Transition((), (1, 0), (2, 2)).iterate(3)
    >>> loop.state, loop.rounds
    (Dim(2, 2), 3)
    >>> (loop >> loop.transition.iterate(5)).rounds
    8
    >>> loop >> Transition((), (1, 0), (4, 4)).iterate(1)
    Traceback (most recent call last):
        ...
    ValueError: two iterations of different transitions do not compose.
    >>> Transition((), (1, 0), (2, 2)).iterate(-1)
    Traceback (most recent call last):
        ...
    ValueError: -1
    """

    transition: Transition
    rounds: int

    def __post_init__(self):
        if self.rounds < 0:
            raise ValueError(self.rounds)

    @property
    def state(self) -> Dim:
        """ The state object, the transition's. """
        return self.transition.state

    def __rshift__(self, other: Iteration) -> Iteration:
        if not isinstance(other, Iteration):
            return NotImplemented
        if other.transition != self.transition:
            raise ValueError(
                "two iterations of different transitions do not compose.")
        return Iteration(self.transition, self.rounds + other.rounds)


def from_map(cmap, inject: bool = True) -> Transition:
    """
    The transition an interpreted map computes in one round.

    Read-only: it reads the boxes, the involution and the port widths, and
    builds no tensor and no index -- in particular it does not touch
    :attr:`~discopy.neural.CMap._routing`, which is where ``torch`` enters.

    Parameters:
        cmap : The :class:`~discopy.neural.CMap` to read.
        inject : Whether the round re-adds the initial messages.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.functor import Interpretation, interpret
    >>> from discopy.neural.skeleton import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> wiring = interpret(Interpretation(
    ...     {peer: Dim(3), state: Dim(5)}, {"cell": None}),
    ...     from_relation(((1, ), (0, )), node))
    >>> round_ = from_map(wiring.cmap, inject=False)
    >>> round_.width == wiring.router.total, round_.is_involution()
    (True, True)
    >>> [spec.name for spec in round_.local]
    ['cell', 'cell']
    """
    ports = cmap.ports
    return Transition(
        local=tuple(interaction_spec(box) for box in cmap.boxes),
        routing=tuple(cmap.edges),
        widths=tuple(cmap.port_widths),
        inputs=tuple(i for i, port in enumerate(ports)
                     if port.kind == PortKind.INPUT),
        outputs=tuple(i for i, port in enumerate(ports)
                      if port.kind == PortKind.OUTPUT),
        inject=inject)
