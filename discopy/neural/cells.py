# -*- coding: utf-8 -*-

"""
The two module shapes a site of a map can carry.

A box of a skeleton is a promise -- its :class:`Signature` says how many
ports it has and which of them are one orbit under a group -- and a cell is
a torch module that keeps the promise.  There are two shapes:

* :class:`Site` : a stateful node.  It encodes every incoming message
  against its own state, pools the encodings over the orbit, runs a
  recurrent cell from the pool and its traced inputs, and broadcasts one
  fresh belief back to every port of the orbit.  Pooling and broadcasting
  are what make it equivariant: permute its message ports and its outputs
  permute with them.
* :class:`Relation` : a stateless hyperedge.  It embeds each member's
  message, pools the embeddings into a summary of the whole relation, and
  answers *each* member with that summary alongside its own message -- an
  equivariant emission rather than an invariant broadcast, which is what
  lets one box speak for a constraint over nine variables instead of the
  thirty-six wires a clique needs to say the same thing.

Both read and write through :meth:`Signature.slices`, so no port offset is
ever written by hand and the cursor arithmetic of a module cannot drift
from the type of its box.  Both derive the arity of their orbit from the
width they are handed, so one shared module serves sites of different
degree -- which is what makes a batch of differently shaped problems one
map (see :mod:`discopy.neural.batch`).

Note
----
A learned cell is only **laxly** structured.  Pool symmetrically and
permutation-equivariance holds, up to the reordering of a floating-point
reduction; :func:`~discopy.neural.signature.check_equivariant` measures
that residual.  Nothing here makes a learned :class:`Relation` satisfy
Frobenius fusion or speciality, and it generally does not:
:func:`fusion_residual` measures how far it is from fusing, and the honest
answer is "not zero".

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:

    Mode
    Site
    Relation
"""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

import torch

from discopy.neural.signature import Signature

#: The order-invariant reductions a cell can pool an orbit with.  Which one
#: is a real architectural choice -- a mean keeps a cell's input scale
#: independent of its degree, a sum keeps it additive over members -- so it
#: is named rather than hidden.
POOL = {
    "mean": lambda encoded: encoded.mean(1),
    "sum": lambda encoded: encoded.sum(1),
}

#: The recurrent cells a :class:`Site` can carry, by name.  A cell with one
#: state takes and returns a tensor, one with two takes and returns a pair.
RECURRENT = {
    "gru": torch.nn.GRUCell,
    "lstm": torch.nn.LSTMCell,
    "rnn": torch.nn.RNNCell,
}


class Mode(StrEnum):
    """
    What a :class:`Site` does with a traced role.

    * :attr:`STATE` : carried and updated by the recurrent cell, emitted
      as its new value.  Several state roles are the several states of one
      cell, e.g. the ``h`` and ``c`` of an ``LSTMCell``.
    * :attr:`INPUT` : read and fed to the recurrent cell, emitted as zeros
      -- so it must be re-injected every round -- or echoed back when the
      site is resumable, so that a run carries its own inputs and can be
      stopped and restarted.
    * :attr:`CARRY` : read and fed to the recurrent cell, emitted
      unchanged.  The site reads it but never writes it; an outer loop
      does.
    """

    STATE = "state"
    INPUT = "input"
    CARRY = "carry"


class Cell(torch.nn.Module):
    """
    What the two cells share: a signature, the widths of its roles, and the
    arithmetic that turns a flat message vector into named blocks.

    The arity of the first orbit is read off the width of the input rather
    than fixed at construction, so one module fills sites of different
    degree; everything else comes from the signature.

    Parameters:
        signature : The signature of the site.
        widths : The width each atomic role carries.
    """
    def __init__(self, signature: Signature, widths: Mapping):
        super().__init__()
        self.signature, self.widths = signature, dict(widths)
        orbit = signature.orbits[0]
        self.orbit = orbit.role
        self.leg = sum(self.widths[atom] for atom in orbit.role)
        self.fixed = signature.width(self.widths) \
            - orbit.copies * orbit.arity * self.leg
        self._places: dict = {}

    def places(self, width: int) -> tuple[int, dict]:
        """
        The arity of the first orbit at this input width, and where each
        role sits in the flat message vector, cached per width.

        Parameters:
            width : The width of the incoming flat message vector.
        """
        if width not in self._places:
            span = width - self.fixed
            arity, remainder = divmod(span, self.leg)
            if remainder or arity <= 0:
                raise ValueError(
                    f"cannot read {width} as a {type(self).__name__} with "
                    f"{self.fixed} fixed and legs of {self.leg}")
            resized = self.signature.resize(self.orbit[0], arity)
            self._places[width] = (arity, resized.slices(self.widths))
        return self._places[width]

    def roles(self, mode: Mode = None) -> tuple:
        """ The atomic roles of the orbits after the first, in order. """
        return tuple(
            atom for orbit in self.signature.orbits[1:] for atom in orbit.role
            if self.widths[atom] and (mode is None or self.mode[atom] == mode))


class Site(Cell):
    """
    A stateful node: the shared cell box of a message-passing solver.

    Each round it encodes every incoming message against its own state,
    pools the encodings over its orbit, runs a recurrent cell from the pool
    and its traced inputs, normalises the result and broadcasts one fresh
    belief to every port of the orbit.  Its traced roles are emitted
    according to their :class:`Mode`.

    Parameters:
        signature : The signature of the site.  Its first orbit is the
                    message orbit; the rest are traced roles.
        widths : The width each atomic role carries; a role of width zero
                 is erased, exactly as ``Dim(0)`` erases its ports.
        mode : What the site does with each traced role.
        hidden : The width of the hidden layers.
        depth : The number of linear layers of the encoder.
        pool : The key in :data:`POOL` the orbit is pooled with.
        recurrent : The key in :data:`RECURRENT` of the update cell.
        emit : Whether to read the belief off the state through a learned
               linear map, or to broadcast the state itself.
        resumable : Whether to echo the :attr:`Mode.INPUT` roles rather
                    than emitting zeros.

    Note
    ----
    The submodules are built in the order ``encode``, ``update``, ``norm``,
    ``emit`` and this order is load-bearing: every constructor draws from
    the global generator, so permuting them permutes every weight in the
    model.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature, Sym
    >>> message, state, clue = Ty("message"), Ty("state"), Ty("clue")
    >>> cell = Site(
    ...     Signature((Orbit(message, 3, Sym.PERM),
    ...                Orbit(state, traced=True), Orbit(clue, traced=True))),
    ...     {message: 4, state: 8, clue: 4},
    ...     {state: Mode.STATE, clue: Mode.INPUT}, hidden=16)
    >>> cell(torch.zeros(2, 3 * 4 + 2 * 8 + 2 * 4)).shape
    torch.Size([2, 36])
    """
    def __init__(self, signature: Signature, widths: Mapping,
                 mode: Mapping, hidden: int, depth: int = 2,
                 pool: str = "mean", recurrent: str = "gru",
                 emit: bool = True, resumable: bool = False):
        super().__init__(signature, widths)
        self.mode = dict(mode)
        self.pooling, self.resumable = POOL[pool], resumable
        self.states = self.roles(Mode.STATE)
        self.inputs = tuple(role for role in self.roles()
                            if self.mode[role] != Mode.STATE)
        if not self.states:
            raise ValueError("a site needs a state to carry")
        self.state_width = self.widths[self.states[0]]
        if any(self.widths[role] != self.state_width
               for role in self.states):
            raise ValueError("the states of one cell must be equally wide")

        self.encode = _mlp(self.state_width + self.leg, hidden, depth)
        self.update = RECURRENT[recurrent](
            hidden + sum(self.widths[role] for role in self.inputs),
            self.state_width)
        self.norm = torch.nn.LayerNorm(self.state_width)
        self.emit = torch.nn.Linear(self.state_width, self.leg) \
            if emit else None

    def forward(self, x):
        arity, places = self.places(x.shape[-1])
        message = x[:, places[self.orbit[0]]].reshape(-1, arity, self.leg)
        carried = [x[:, places[role]] for role in self.states]
        given = {role: x[:, places[role]] for role in self.inputs}

        pooled = self.pooling(self.encode(torch.cat([
            carried[0].unsqueeze(1).expand(-1, arity, -1), message], -1)))
        updated = self.update(
            torch.cat([pooled] + list(given.values()), -1),
            carried[0] if len(carried) == 1 else tuple(carried))
        updated = [updated] if isinstance(updated, torch.Tensor) \
            else list(updated)
        updated[0] = self.norm(updated[0])
        assert updated[0].shape[-1] == self.state_width, \
            "the state changed width"

        signal = updated[0] if self.emit is None else self.emit(updated[0])
        belief = signal.unsqueeze(1).expand(-1, arity, -1).reshape(
            -1, arity * self.leg)
        emitted = dict(zip(self.states, updated))
        for role, value in given.items():
            emitted[role] = value if (
                self.mode[role] == Mode.CARRY or self.resumable) \
                else torch.zeros_like(value)
        out = torch.cat([belief] + _emissions(
            self.signature, self.widths, emitted), -1)
        assert out.shape == x.shape, "the cell changed its port widths"
        return out


class Relation(Cell):
    """
    A stateless hyperedge: the shared relation box of a factor graph.

    A permutation-equivariant Deep-Sets relation over the members of a
    constraint: it embeds each incoming message, pools the embeddings into
    an order-invariant summary of the relation, and answers each member
    with that summary alongside its own message.

    Parameters:
        signature : The signature of the site, one orbit of members.
        widths : The width each atomic role carries.
        hidden : The width of the hidden layers.
        pool : The key in :data:`POOL` the members are pooled with.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature, Sym
    >>> message = Ty("message")
    >>> unit = Signature((Orbit(message, 9, Sym.PERM), ))
    >>> box = Relation(unit, {message: 4}, hidden=8)
    >>> box(torch.zeros(2, 36)).shape
    torch.Size([2, 36])

    The equations of its signature hold, up to the reordering of a
    floating-point sum:

    >>> from discopy.neural import check_equivariant
    >>> check_equivariant(box.double(), unit, {message: 4})[message] < 1e-12
    True
    """
    def __init__(self, signature: Signature, widths: Mapping, hidden: int,
                 pool: str = "sum"):
        super().__init__(signature, widths)
        if len(signature.orbits) != 1:
            raise ValueError("a relation has a single orbit of members")
        self.pooling = POOL[pool]
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.leg, hidden), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(self.leg + hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, self.leg))

    def forward(self, x):
        arity, _ = self.places(x.shape[-1])
        message = x.reshape(-1, arity, self.leg)
        pooled = self.pooling(self.phi(message)).unsqueeze(1).expand(
            -1, arity, -1)
        out = self.rho(torch.cat([message, pooled], -1))
        assert out.shape == message.shape, "the relation changed its widths"
        return out.reshape(-1, arity * self.leg)


def _mlp(dom: int, hidden: int, depth: int) -> torch.nn.Sequential:
    """ ``depth`` linear layers from ``dom`` to ``hidden``, ReLU between. """
    layers: list = []
    for index in range(depth):
        if index:
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden if index else dom, hidden))
    return torch.nn.Sequential(*layers)


def _emissions(signature: Signature, widths: Mapping, values: Mapping) -> list:
    """
    What a site writes on the orbits after the first, in port order: each
    surviving role once per copy of its leg.
    """
    return [values[atom]
            for orbit in signature.orbits[1:]
            for _ in range(orbit.copies * orbit.arity)
            for atom in orbit.role if widths[atom]]


def fusion_residual(module, signature: Signature, widths: Mapping,
                    arity: int = 4, rounds: int = 4, batch: int = 4,
                    seed: int = 0) -> float:
    """
    How far a learned relation is from fusing, i.e. from being a spider.

    The Frobenius fusion law says two spiders wired along a leg are one
    spider over the remaining legs.  Here the two relations are glued
    along their last leg and message passing is run on the composite until
    the shared wire settles; the free legs are then compared against one
    relation over all of them.  A learned module does **not** satisfy the
    law -- it is lax, not strict -- so this is the number that says so,
    reported rather than a docstring claiming a structure the weights do
    not have.

    Parameters:
        module : The relation to measure.
        signature : Its signature, one orbit of members.
        widths : The width each atomic role carries.
        arity : The number of free legs on each side of the gluing.
        rounds : The exchanges along the shared wire.
        batch : The number of random rows to measure on.
        seed : The seed of the random input.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature, Sym
    >>> message = Ty("message")
    >>> unit = Signature((Orbit(message, 4, Sym.PERM), ))
    >>> box = Relation(unit, {message: 2}, hidden=4).double()
    >>> fusion_residual(box, unit, {message: 2}) > 1e-3
    True
    """
    leg = sum(widths[atom] for atom in signature.orbits[0].role)
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(batch, 2 * arity * leg, generator=generator,
                       dtype=torch.double)
    left, right = rows[:, :arity * leg], rows[:, arity * leg:]
    shared = torch.zeros(batch, 2 * leg, dtype=rows.dtype)
    with torch.no_grad():
        for _ in range(rounds):
            one = module(torch.cat([left, shared[:, leg:]], -1))
            other = module(torch.cat([right, shared[:, :leg]], -1))
            shared = torch.cat([one[:, -leg:], other[:, -leg:]], -1)
        glued = torch.cat([one[:, :-leg], other[:, :-leg]], -1)
        return float((glued - module(rows)).abs().max())
