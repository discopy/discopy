# -*- coding: utf-8 -*-

"""
A functor from diagrams to runnable maps, as a torch module.

:class:`MapNN` holds the width of every role and the shared module of every
generator, compiles each diagram it is given once through
:func:`~discopy.neural.map.interpret`, builds an initial state out of the
caller's inputs and runs the rounds of :meth:`~discopy.neural.CMap.forward`
on it. A state is one flat tensor of shape ``(rows, total)``, the messages
of every port in port order, that :meth:`MapNN.read` and :meth:`MapNN.write`
address by ``(generator name, role)`` rather than by offset, through the
:func:`~discopy.neural.map.families` of the diagram, so no port arithmetic is
ever written by hand::

    loss = sum(criterion(readout(model.read(diagram, s, answer)), target)
               for s in model(diagram, {clue: x}, deep=True))

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    MapNN
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

from discopy.neural.map import families, interpret


class MapNN(torch.nn.Module):
    """
    A functor from diagrams to runnable maps, as a torch module: the width
    of every role, the shared module of every generator, and how many
    rounds to run.

    Parameters:
        ob : The :class:`~discopy.neural.Dim` each atomic role carries.
        ar : The torch module filling each generator name, shared by every
             site of that name.
        rounds : The rounds of message passing one call performs.
        inject : Whether every round re-adds the initial messages, i.e.
                 whether the transition is :math:`\\sigma(\\Phi(s)) + i`.
        cache : How many compiled diagrams to keep, least recently used
                first out.

    Note
    ----
    :meth:`compile` deliberately shadows ``torch.nn.Module.compile``, which
    is ``torch.compile(self)``: here the word means *compiling a diagram
    into a map*, which is what this class is for.  The ``torch.compile``
    of the per-round step is :meth:`compile_rounds`.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature  # doctest: +EXTRA
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
    <torch...>
    >>> model = MapNN(ob={peer: Dim(3), state: Dim(4)},
    ...               ar={"cell": torch.nn.PReLU()}, rounds=3)

    One model, two shapes, one set of weights:

    >>> pair = from_relation(((1, ), (0, )), node)
    >>> path = from_relation(((1, ), (0, 2), (1, )), node)
    >>> [model(shape).shape for shape in (pair, path)]
    [torch.Size([1, 22]), torch.Size([1, 36])]
    >>> model.read(path, model(path), ("cell", state)).shape
    torch.Size([1, 3, 4])
    """
    def __init__(self, ob: Mapping, ar: Mapping, rounds: int = 1,
                 inject: bool = False, cache: int = 128):
        super().__init__()
        self.ob = dict(ob)
        self.ar = torch.nn.ModuleDict(ar)
        self.rounds, self.inject = rounds, inject
        self.cache = cache
        self.hits, self.misses = 0, 0
        self.compiled: OrderedDict = OrderedDict()
        self.rounds_kwargs: dict = None

    def compile(self, diagram) -> tuple:
        """
        The map a diagram means under this interpretation and the
        :func:`~discopy.neural.map.families` of its ports, cached by the
        identity of the diagram.

        Every call is counted in :attr:`hits` or :attr:`misses`, since
        compiling is the expensive half of running a diagram once and free
        every time after: a model whose diagrams do not fit its
        :attr:`cache` recompiles them every epoch, which is a wall clock a
        loss curve cannot show.  See :meth:`cache_stats`.

        The cache pins each diagram beside its map, so that a key can never
        be a recycled ``id``, and is keyed by the diagram alone: a model
        whose modules are swapped afterwards is a new model.

        Parameters:
            diagram : A closed diagram or map in the source category, or a
                      :class:`~discopy.neural.Batch` of them.
        """
        key = diagram.cache_key() if hasattr(diagram, "cache_key") \
            else id(diagram)
        if key in self.compiled:
            self.hits += 1
            self.compiled.move_to_end(key)
            return self.compiled[key][1:]
        self.misses += 1
        source = getattr(diagram, "diagram", diagram)
        cmap = interpret(source, self.ob, dict(self.ar))
        if self.rounds_kwargs is not None:
            cmap.compile(**self.rounds_kwargs)
        ports, heads = families(source, cmap, self.ob)
        self.compiled[key] = (diagram, cmap, ports, heads)
        while len(self.compiled) > self.cache:
            self.compiled.popitem(last=False)
        return cmap, ports, heads

    def cache_stats(self, reset: bool = False) -> dict:
        """
        What :meth:`compile` has done so far: ``hits``, ``misses``, how
        many maps are ``held`` and the ``capacity`` they are held in.  A
        ``held`` equal to ``capacity`` with ``misses`` still rising is an
        evicting cache, i.e. a recompilation per epoch.

        Parameters:
            reset : Whether to zero the counters, so that a caller can
                    measure one epoch rather than a whole run.

        Example
        -------
        >>> model = MapNN({}, {}, cache=1)
        >>> model.cache_stats()
        {'hits': 0, 'misses': 0, 'held': 0, 'capacity': 1}
        """
        found = {"hits": self.hits, "misses": self.misses,
                 "held": len(self.compiled), "capacity": self.cache}
        if reset:
            self.hits, self.misses = 0, 0
        return found

    def compile_rounds(self, **kwargs) -> MapNN:
        """
        Compile the per-round step of every map with ``torch.compile``; see
        :meth:`discopy.neural.CMap.compile`.

        Parameters:
            kwargs : Passed through to ``torch.compile``.
        """
        self.rounds_kwargs = kwargs
        for _, cmap, _, _ in self.compiled.values():
            cmap.compile(**kwargs)
        return self

    def initial(self, diagram, values: Mapping = None, rows: int = None,
                like=None):
        """
        The initial flat state: the given values on every copy of their
        family, zeros everywhere else.

        Parameters:
            diagram : The diagram to run.
            values : A tensor of shape ``(rows, sites, width)`` per
                     ``(generator name, role)`` family.
            rows : The batch size, read off the values by default.
            like : A tensor whose dtype and device the state follows, read
                   off the values or off the parameters by default.
        """
        cmap, _, _ = self.compile(diagram)
        values = dict(values or {})
        reference = next(iter(values.values()), None)
        if reference is None:
            reference = next(iter(self.parameters()), None)
        elif rows is None:
            rows = len(reference)
        like = reference if like is None else like
        state = cmap.zeros(1 if rows is None else rows, like=like)
        for key, value in values.items():
            state = self.write(diagram, state, key, value)
        return state

    def read(self, diagram, state, key, every: bool = False):
        """
        The messages of a family, of shape ``(rows, sites, width)``.

        Parameters:
            diagram : The diagram the state belongs to.
            state : The flat messages.
            key : A ``(generator name, role)`` pair.
            every : Whether to read every port rather than one per traced
                    pair.
        """
        cmap, ports, heads = self.compile(diagram)
        return cmap.read(state, (ports if every else heads)[key])

    def write(self, diagram, state, key, values):
        """
        A copy of the state with values written on every copy of a family,
        one value per site broadcast to each copy of its trace.

        Parameters:
            diagram : The diagram the state belongs to.
            state : The flat messages.
            key : A ``(generator name, role)`` pair.
            values : A tensor of shape ``(rows, sites, width)``.
        """
        cmap, ports, heads = self.compile(diagram)
        copies = len(ports[key]) // len(heads[key])
        if copies > 1:
            values = values.repeat_interleave(copies, dim=1)
        return cmap.write(state, ports[key], values)

    def sites(self, diagram, key) -> int:
        """
        How many heads a family has in a diagram, i.e. how many values
        :meth:`read` returns for it: one per traced pair, and one per port
        of an untraced role.

        Parameters:
            diagram : The diagram to compile.
            key : A ``(generator name, role)`` pair.
        """
        return len(self.compile(diagram)[2][key])

    def forward(self, diagram, init=None, deep: bool = False,
                rounds: int = None, inject: bool = None):
        """
        The flat state after the rounds of message passing, or the list of
        the states after every round under ``deep``.

        Parameters:
            diagram : The diagram to run.
            init : The initial values per family, as in :meth:`initial`, or
                   a flat state to start from.
            deep : Whether to return the state after every round.
            rounds : The rounds of message passing, :attr:`rounds` by
                     default.
            inject : Whether to re-add the initial state after every round,
                     :attr:`inject` by default.
        """
        state = init if isinstance(init, torch.Tensor) \
            else self.initial(diagram, init)
        cmap, _, _ = self.compile(diagram)
        return cmap(
            init=state, n_rounds=self.rounds if rounds is None else rounds,
            inject=self.inject if inject is None else inject,
            return_rounds=deep, return_flat=True)
