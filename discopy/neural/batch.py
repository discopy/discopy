# -*- coding: utf-8 -*-

"""
Batching over variable structure, as the monoidal product of maps.

A batch of problems with the *same* shape is a leading tensor axis and
needs nothing from the formalism.  A batch of problems with *different*
shapes -- grids of different sizes, graphs with different numbers of nodes
-- is a different question, and the formalism already answers it: the
disjoint union of maps is their monoidal product, ``a @ b``, and running
the product is running both at once.

Nothing is lost by doing so.  :attr:`~discopy.neural.CMap._fused_routing`
groups boxes by the module they share, so a site of ``a`` and a site of
``b`` with the same module and the same port widths land in one group and
cost one batched call between them; only sites of genuinely different
degree cost a call of their own.  And because the product of closed maps
lays each member's ports out contiguously and in order, the flat state of
the product *is* the concatenation of the members' flat states, so
:meth:`Batch.join` and :meth:`Batch.split` are a concatenation and a split.

Two caches keep this cheap.  The product map is built once per *sequence
of member instances* -- so intern the skeletons, one instance per shape,
as the ``lru_cache`` builders of a task naturally do, and the same mix of
shapes in the same order hits the cache; and :func:`bucket` rounds the
member count up to a coarse ladder by repeating the last member, so a run
sees a handful of distinct shapes rather than one per batch and
``torch.compile`` does not recompile for each.

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

import torch

from discopy.neural.functor import Interpretation, interpret

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
    Several skeletons run as one map: their monoidal product.

    Parameters:
        parts : The skeletons to run together, one per problem.
        interpretation : The widths and the modules, shared by all of them
                         -- sharing a module is what makes the batch one
                         call per group rather than one call per member.
        pad : Whether to pad the member count up to a :func:`bucket` by
              repeating the last member, whose result :meth:`split` drops.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.skeleton import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> meaning = Interpretation(
    ...     {peer: Dim(3), state: Dim(5)}, {"cell": torch.nn.Identity()})
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> quad = from_relation(((1, ), (0, ), (3, ), (2, )), node)
    >>> together = Batch((pair, quad), meaning)
    >>> len(together.wiring.cmap.boxes), together.totals
    (6, (26, 52))

    One module call covers every site of every member, rather than one
    call per member:

    >>> [n_boxes for _, _, n_boxes, _
    ...  in together.wiring.cmap._fused_routing["metas"]]
    [6]
    >>> together.split(together.join([
    ...     torch.zeros(1, 26), torch.ones(1, 52)]))[1].mean()
    tensor(1.)
    """
    #: The interpreted products already built, keyed by the identity of
    #: the interpretation and of each member; the entry pins the members,
    #: so a key can never be a recycled ``id``.
    cache: dict = {}

    def __init__(self, parts, interpretation: Interpretation,
                 pad: bool = False):
        parts = tuple(parts)
        if not parts:
            raise ValueError("a batch needs at least one member")
        self.given = len(parts)
        if pad:
            parts += (parts[-1], ) * (bucket(len(parts)) - len(parts))
        self.parts, self.interpretation = parts, interpretation
        self.totals = tuple(self.width(part) for part in parts)
        key = (id(interpretation), tuple(id(part) for part in parts))
        if key not in Batch.cache:
            Batch.cache[key] = (interpretation, parts, interpret(
                interpretation, reduce(matmul, parts)))
        self.wiring = Batch.cache[key][-1]

    def width(self, part) -> int:
        """
        The flat width of one member, without building its map: the sum
        over its boxes of the widths of the roles that survive.

        Parameters:
            part : The skeleton of the member.
        """
        widths = self.interpretation.widths
        return sum(widths[role] for index in range(len(part.boxes))
                   for role in part.signature(index).roles)

    def join(self, states) -> "torch.Tensor":
        """
        The flat state of the product, from one flat state per member.

        The product of closed maps lays each member's ports out
        contiguously and in order, so this is a concatenation.

        Parameters:
            states : One tensor per member, in member order.
        """
        states = list(states)
        if len(states) != self.given:
            raise ValueError(f"expected {self.given} states")
        states += [states[-1]] * (len(self.parts) - self.given)
        return torch.cat(states, -1)

    def split(self, flat) -> list:
        """
        The flat state of each member, from the flat state of the product;
        padding members are dropped.

        Parameters:
            flat : The flat state of the product.
        """
        return list(torch.split(flat, list(self.totals), -1))[:self.given]
