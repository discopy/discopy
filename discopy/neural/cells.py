# -*- coding: utf-8 -*-

"""
The concrete neural interpretations a generator can carry, one per symmetry
a signature can declare.

A generator of a diagram is a promise -- its
:class:`~discopy.neural.Signature` says how many ports it has and which of
them are one orbit under a group -- and a cell is a torch module that keeps
the promise.  These are the modules a :class:`~discopy.neural.MapNN` is
given as its ``ar``; one instance is shared by every site of its name,
which is what makes the model size independent of the diagram size.  The
two :attr:`Sym.PERM` shapes:

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

and the two shapes for the other symmetries a signature can declare:

* :class:`Gate` : the :attr:`Sym.NONE` cell, distinguishable ports.  One
  MLP over the whole concatenated port block, no pooling and no weight
  sharing across ports, hence no equation to keep.
* :class:`Cyclic` : the :attr:`Sym.CYCLIC` cell, a planar node.  The
  circular convolution of its legs -- one linear kernel per offset, i.e.
  one linear map on the rotation-rolled block -- so rotating its inputs
  rotates its outputs exactly, but an arbitrary permutation does not
  commute with it.

Pooling is what buys degree independence, so the symmetric cells serve
sites of any degree while :class:`Gate` and :class:`Cyclic` are
arity-fixed by construction: a module with one weight per port, or one
kernel entry per offset, cannot pretend to a degree it was not built at.

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
reduction; :func:`~discopy.neural.laws.check_equivariant` measures that
residual.  Nothing here makes a learned :class:`Relation` satisfy Frobenius
fusion or speciality, and it generally does not:
:func:`~discopy.neural.laws.fusion_residual` measures how far it is from
fusing, and the honest answer is "not zero".

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:

    Mode
    Site
    Relation
    Gate
    Cyclic
"""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

import torch

from discopy.neural.signature import Signature

#: The order-invariant reductions a cell can pool an orbit with.  Which one
#: is a real architectural choice -- a mean keeps a cell's input scale
#: independent of its degree, a sum keeps it additive over members, a max
#: keeps an *extremum* of them and so is the only one of the three whose
#: value stays inside their range however many there are: a sum grows with
#: the members and a mean divides by them, while adding a member a max
#: dominates changes nothing at all.  So it is named rather than hidden.
POOL = {
    "mean": lambda encoded: encoded.mean(1),
    "sum": lambda encoded: encoded.sum(1),
    "max": lambda encoded: encoded.amax(1),
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
        per_leg : Whether the emission answers each leg separately,
                  ``out_i = emit(state || m_i)``, rather than
                  broadcasting one belief to every leg.  Still
                  permutation-equivariant -- the state is pooled, each
                  answer reads its own message -- and strictly more
                  general than the broadcast, which a heavily
                  over-constrained graph gets away with but a sparse
                  pairwise one does not, having no relation boxes to
                  carry the asymmetry.  Requires ``emit``.
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
    >>> message, state, given = Ty("message"), Ty("state"), Ty("given")
    >>> cell = Site(
    ...     Signature((Orbit(message, 3, Sym.PERM),
    ...                Orbit(state, traced=True), Orbit(given, traced=True))),
    ...     {message: 4, state: 8, given: 4},
    ...     {state: Mode.STATE, given: Mode.INPUT}, hidden=16)
    >>> cell(torch.zeros(2, 3 * 4 + 2 * 8 + 2 * 4)).shape
    torch.Size([2, 36])
    """
    def __init__(self, signature: Signature, widths: Mapping,
                 mode: Mapping, hidden: int, depth: int = 2,
                 pool: str = "mean", recurrent: str = "gru",
                 emit: bool = True, per_leg: bool = False,
                 resumable: bool = False):
        super().__init__(signature, widths)
        if per_leg and not emit:
            raise ValueError("per-leg emission needs a learned emit map")
        self.mode, self.per_leg = dict(mode), per_leg
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
        self.emit = torch.nn.Linear(
            self.state_width + (self.leg if per_leg else 0), self.leg) \
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

        if self.per_leg:
            belief = self.emit(torch.cat([
                updated[0].unsqueeze(1).expand(-1, arity, -1), message],
                -1)).reshape(-1, arity * self.leg)
        else:
            signal = updated[0] if self.emit is None else \
                self.emit(updated[0])
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


class Gate(Cell):
    """
    A fixed-arity box with distinguishable ports: the :attr:`Sym.NONE`
    cell.

    One MLP over the whole concatenated port block, reading and writing
    every port of the signature at once.  No pooling and no weight
    sharing across ports means no equation to keep -- this is the least
    symmetric cell, the one a :attr:`Sym.NONE` signature admits -- and no
    cross-degree sharing either: a module with one weight per port is
    arity-fixed by construction, which is the honest cost of dropping
    the symmetry.

    A two-port gate is how an operation lands *on a wire* rather than in
    the engine -- a negation between a literal and its clause, say -- and
    once it is a box, its equations become measurable properties of the
    module: whether two gates in series are the identity is a number,
    not a docstring.

    Parameters:
        signature : The signature of the site; every orbit should be
                    :attr:`Sym.NONE`, since an MLP keeps no other promise.
        widths : The width each atomic role carries.
        hidden : The width of the hidden layers.
        depth : The number of linear layers before the output one.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature
    >>> wire = Ty("wire")
    >>> box = Gate(Signature((Orbit(wire, 2), )), {wire: 3}, hidden=8)
    >>> box(torch.zeros(2, 6)).shape
    torch.Size([2, 6])
    """
    def __init__(self, signature: Signature, widths: Mapping, hidden: int,
                 depth: int = 2):
        super().__init__(signature, widths)
        self.width = signature.width(self.widths)
        self.mlp = torch.nn.Sequential(
            _mlp(self.width, hidden, depth), torch.nn.ReLU(),
            torch.nn.Linear(hidden, self.width))

    def forward(self, x):
        if x.shape[-1] != self.width:
            raise ValueError(
                f"a Gate is arity-fixed: expected width {self.width}, "
                f"got {x.shape[-1]}")
        return self.mlp(x)


class Cyclic(Cell):
    """
    A planar node: the :attr:`Sym.CYCLIC` cell, the circular convolution
    of its legs with a kernel indexed by offset.

    ``out[i] = rho(phi(m[i] || m[i+1] || ... || m[i+d-1]))`` with indices
    mod ``d``: one linear kernel per offset is one linear map on the
    rotation-rolled block, so rotating the inputs rotates the outputs
    *exactly* -- the same arithmetic lands at the shifted position -- while
    an arbitrary permutation scrambles the offsets and does not commute.
    Like :class:`Gate` it is arity-fixed: the kernel has one entry per
    offset, so there is nothing to share across degrees.

    Parameters:
        signature : The signature of the site, one cyclic orbit of legs.
        widths : The width each atomic role carries.
        hidden : The width of the hidden layer.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature, Sym
    >>> from discopy.neural import check_equivariant
    >>> leg = Ty("leg")
    >>> planar = Signature((Orbit(leg, 5, Sym.CYCLIC), ))
    >>> box = Cyclic(planar, {leg: 3}, hidden=8).double()
    >>> check_equivariant(box, planar, {leg: 3})[leg] == 0.0
    True
    """
    def __init__(self, signature: Signature, widths: Mapping, hidden: int):
        super().__init__(signature, widths)
        if len(signature.orbits) != 1:
            raise ValueError("a cyclic cell has a single orbit of legs")
        self.arity = signature.orbits[0].arity
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.arity * self.leg, hidden), torch.nn.ReLU())
        self.rho = torch.nn.Linear(hidden, self.leg)
        offsets = torch.arange(self.arity)
        self.register_buffer(
            "roll", (offsets.unsqueeze(1) + offsets) % self.arity,
            persistent=False)

    def forward(self, x):
        arity, _ = self.places(x.shape[-1])
        if arity != self.arity:
            raise ValueError(
                f"a Cyclic cell is arity-fixed: built at {self.arity}, "
                f"read {arity}")
        message = x.reshape(-1, arity, self.leg)
        rolled = message[:, self.roll].reshape(-1, arity, arity * self.leg)
        out = self.rho(self.phi(rolled))
        assert out.shape == message.shape, "the cell changed its widths"
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
