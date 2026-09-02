# -*- coding: utf-8 -*-

"""
Batching over heterogeneous diagrams, as the monoidal product of maps.

A batch of samples whose diagrams have the *same* shape is a leading tensor
axis and needs nothing from the formalism.  A batch of samples with
*different* shapes -- grids of different sizes, graphs with different
numbers of nodes, sentences of different lengths -- is a different
question, and the formalism already answers it: the disjoint union of maps
is their monoidal product ``a @ b``, and running the product is running
both at once.

Nothing is lost by doing so.  :attr:`~discopy.neural.CMap.routing`
groups boxes by the module they share, so a site of ``a`` and a site of
``b`` with the same module and the same port widths land in one group and
cost one batched call between them; only sites of genuinely different
degree cost a call of their own.  And because the product of closed maps
lays each member's ports out contiguously and in order, the flat state of
the product *is* the concatenation of the members' flat states, so
:meth:`Batch.join` and :meth:`Batch.split_state` are a concatenation and a
split.

Two habits keep this cheap.  Intern the diagrams -- one instance per shape,
as an ``lru_cache`` builder naturally gives -- so that
:meth:`~discopy.neural.MapNN.compile` hits its cache; and pass ``pad=True``
so that :func:`bucket` rounds the member count up to a coarse ladder by
repeating the last member, and a run sees a handful of distinct shapes
rather than one per batch.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:

    Batch
"""

from __future__ import annotations

from functools import reduce
from operator import matmul

from discopy.neural.core import box_ports

#: The member counts a batch is rounded up to, so that a run sees a few
#: distinct shapes rather than one per batch.
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def bucket(size: int, ladder=BUCKETS) -> int:
    """
    The smallest bucket at least as large as ``size``, or ``size`` itself
    when it is past the end of the ladder.

    Parameters:
        size : The number of members in the batch.
        ladder : The bucket sizes, increasing.

    Example
    -------
    >>> [bucket(n) for n in (1, 3, 5, 9, 2000)]
    [1, 4, 8, 16, 2000]
    """
    return next((step for step in ladder if step >= size), size)


class Batch:
    """
    Several diagrams run as one: their monoidal product.

    A batch is itself a diagram as far as
    :meth:`~discopy.neural.MapNN.compile` is concerned, so a model runs it
    exactly as it runs a single sample; what a batch adds is the
    bookkeeping to cut the result back into members.

    Parameters:
        parts : The diagrams to run together, one per sample.
        pad : Whether to pad the member count up to a :func:`bucket` by
              repeating the last member, whose result :meth:`split` drops.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> quad = from_relation(((1, ), (0, ), (3, ), (2, )), node)
    >>> together = Batch((pair, quad))
    >>> len(together.diagram.boxes), together.sizes(("cell", state))
    (6, (2, 4))
    >>> together.widths({peer: 3, state: 5})
    (26, 52)
    """
    def __init__(self, parts, pad: bool = False):
        parts = tuple(parts)
        if not parts:
            raise ValueError("a batch needs at least one member")
        self.given = len(parts)
        if pad:
            parts += (parts[-1], ) * (bucket(len(parts)) - len(parts))
        self.parts = parts
        self._diagram = None

    def __len__(self) -> int:
        return self.given

    @property
    def diagram(self):
        """ The monoidal product of the members, built once. """
        if self._diagram is None:
            self._diagram = reduce(matmul, self.parts)
        return self._diagram

    def cache_key(self) -> tuple:
        """
        What :meth:`~discopy.neural.MapNN.compile` keys its cache on: the
        identity of each member, so that a fresh batch of interned diagrams
        hits the cache.
        """
        return ("batch", ) + tuple(id(part) for part in self.parts)

    def sizes(self, key) -> tuple[int, ...]:
        """
        The number of sites of a family in each *given* member, i.e. the
        split sizes of the axis a model's output ranges over.

        Parameters:
            key : A ``(generator name, role)`` pair, whose role must
                  survive the interpretation.
        """
        return tuple(_sites(part, *key) for part in self.parts)[:self.given]

    def widths(self, ob) -> tuple[int, ...]:
        """
        The flat state width of each *given* member, without compiling it.

        Parameters:
            ob : The width each atomic role carries, as an integer or a
                 :class:`~discopy.neural.Dim`.
        """
        def width(role):
            found = ob[role]
            return found if isinstance(found, int) else sum(found.inside)

        return tuple(
            sum(width(role) for box in part.boxes
                for role in tuple(box.dom) + tuple(box.cod))
            for part in self.parts)[:self.given]

    def split(self, values, key) -> list:
        """
        A tensor over the site axis of the product -- logits, targets, a
        correctness mask -- cut into one piece per given member, the
        padding dropped.

        Parameters:
            values : A tensor of shape ``(rows, sites, ...)``.
            key : The ``(generator name, role)`` the site axis ranges over.
        """
        import torch
        return list(torch.split(
            values, list(self._all_sizes(key)), dim=1))[:self.given]

    def join(self, states):
        """
        The flat state of the product, from one flat state per given
        member: a concatenation, because the product lays each member's
        ports out contiguously and in order.

        Parameters:
            states : One tensor per given member, in member order.
        """
        import torch
        states = list(states)
        if len(states) != self.given:
            raise ValueError(f"expected {self.given} states")
        states += [states[-1]] * (len(self.parts) - self.given)
        return torch.cat(states, -1)

    def split_state(self, flat, ob) -> list:
        """
        The flat state of each given member, from that of the product.

        Parameters:
            flat : The flat state of the product.
            ob : The width each atomic role carries.
        """
        import torch
        sizes = tuple(
            sum(_width(ob, role) for box in part.boxes
                for role in tuple(box.dom) + tuple(box.cod))
            for part in self.parts)
        return list(torch.split(flat, list(sizes), -1))[:self.given]

    def _all_sizes(self, key) -> tuple[int, ...]:
        """ :meth:`sizes` over every member, padding included. """
        return tuple(_sites(part, *key) for part in self.parts)


def _width(ob, role) -> int:
    """ The integer width a role carries, given ints or ``Dim``s. """
    found = ob[role]
    return found if isinstance(found, int) else sum(found.inside)


def _sites(cmap, name: str, role) -> int:
    """
    How many sites of a name carry a role in a map: one per leg, counting
    a traced leg once, exactly as the heads returned by
    :func:`~discopy.neural.map.families` do.
    """
    cmap = cmap.to_map() if hasattr(cmap, "to_map") else cmap
    count = 0
    for index, box in enumerate(cmap.boxes):
        if box.name != name:
            continue
        ports = box_ports(cmap, index)
        place = {port: i for i, port in enumerate(ports)}
        for position, found in enumerate(tuple(box.dom) + tuple(box.cod)):
            if found == role and place.get(
                    cmap.edges[ports[position]], position) >= position:
                count += 1
    return count
