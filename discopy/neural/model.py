# -*- coding: utf-8 -*-

"""
:class:`MapNN`: a neural interpretation of DisCoPy diagrams, as an ordinary
torch module.

A :class:`MapNN` is three things and nothing else:

* ``ob`` -- the width each atomic role carries, as a
  :class:`~discopy.neural.Dim`.  Sending a role to ``Dim(0)`` erases its
  ports and the wires on them, which is how one diagram serves two models.
* ``ar`` -- one shared learnable module per generator *name*.  Every site
  of that name in every diagram is the same module, so the model size is
  independent of the diagram size and a dataset of differently shaped
  diagrams trains one set of weights.
* ``solver`` -- how the compiled interaction is executed, see
  :mod:`discopy.neural.solver`.

Given a diagram it compiles the two into a global interaction
:math:`T_{D,\\theta} : S_D \\to S_D` (:class:`~discopy.neural.map.
Interaction`), builds an initial state out of the caller's inputs, hands it
to the solver and returns the states a loss may look at.  The workflow is
then ordinary PyTorch::

    model = MapNN(ob, ar, solver=Iterate(rounds=16))
    for diagram, x, target in loader:
        states = model(diagram, {("cell", clue): encode(x)}, deep=True)
        loss = sum(criterion(readout(model.read(diagram, s, answer)), target)
                   for s in states)
        loss.backward(); optimizer.step()

Nothing about training is here: no loop, no optimizer, no schedule of
learning rates.  A :class:`MapNN` is a :class:`torch.nn.Module`, so
``.parameters()``, ``.to(device)``, ``.state_dict()`` and every optimizer
work as usual.

A dataset of diagrams
---------------------

A sample is a ``(diagram, inputs, target)`` triple, and the diagrams need
not agree: two grid sizes, two graphs, two sentences.  Samples of the
*same* shape batch along the leading tensor axis and need nothing from the
formalism.  Samples of *different* shapes are a
:class:`~discopy.neural.Batch`, i.e. the monoidal product of their maps,
which :meth:`compile` handles like any other diagram.

Compiling a diagram costs a permutation and a handful of index tensors, so
:meth:`compile` caches its result; **intern the diagrams** -- build one per
shape, e.g. behind an ``lru_cache`` -- and a recurring mix of shapes costs
one compilation ever.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:

    MapNN
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

from discopy.neural.map import Interaction, interpret
from discopy.neural.solver import Iterate, Solver


class MapNN(torch.nn.Module):
    """
    A functor from diagrams to runnable maps, as a torch module: the width
    of every role, the shared module of every generator, and a solver.

    Parameters:
        ob : The :class:`~discopy.neural.Dim` each atomic role carries.
        ar : The torch module filling each generator name, shared by every
             site of that name.
        solver : How to run a compiled diagram, ``Iterate()`` by default.
        cache : How many compiled diagrams to keep, least recently used
                first out.

    Note
    ----
    :meth:`compile` deliberately shadows ``torch.nn.Module.compile``, which
    is ``torch.compile(self)``: here the word means *compiling a diagram
    into an interaction*, which is what this class is for.  The
    ``torch.compile`` of the per-round step is :meth:`compile_rounds`.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Mode, Orbit, Signature, Site, Sym
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1, Sym.PERM), Orbit(state, traced=True)))
    >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
    <torch...>
    >>> model = MapNN(
    ...     ob={peer: Dim(3), state: Dim(4)},
    ...     ar={"cell": Site(node, {peer: 3, state: 4},
    ...                      {state: Mode.STATE}, hidden=8)},
    ...     solver=Iterate(rounds=3))

    One model, two shapes, one set of weights:

    >>> pair = from_relation(((1, ), (0, )), node)
    >>> path = from_relation(((1, ), (0, 2), (1, )), node)
    >>> [model(shape).shape for shape in (pair, path)]
    [torch.Size([1, 22]), torch.Size([1, 36])]
    >>> model.read(path, model(path), ("cell", state)).shape
    torch.Size([1, 3, 4])
    """
    def __init__(self, ob: Mapping, ar: Mapping, solver: Solver = None,
                 cache: int = 128):
        super().__init__()
        self.ob = dict(ob)
        self.ar = torch.nn.ModuleDict(ar)
        self.solver = Iterate() if solver is None else solver
        self.cache = cache
        self.hits, self.misses = 0, 0
        self._compiled: OrderedDict = OrderedDict()
        self._rounds: dict = None

    # --- syntax to semantics ------------------------------------------------

    def compile(self, diagram) -> Interaction:
        """
        The global interaction a diagram means under this interpretation,
        cached by the identity of the diagram.

        Every call is counted in :attr:`hits` or :attr:`misses`, since
        compiling is the expensive half of running a diagram once and free
        every time after: a model whose diagrams do not fit its
        :attr:`cache` recompiles them every epoch, which is a wall clock a
        loss curve cannot show.  See :meth:`cache_stats`.

        Parameters:
            diagram : A closed diagram or map in the source category, a
                      :class:`~discopy.neural.Batch` of them, or an
                      :class:`~discopy.neural.map.Interaction` already, in
                      which case it is returned unchanged.
        """
        if isinstance(diagram, Interaction):
            return diagram
        key = diagram.cache_key() if hasattr(diagram, "cache_key") \
            else id(diagram)
        if key in self._compiled:
            self.hits += 1
            self._compiled.move_to_end(key)
            return self._compiled[key][1]
        self.misses += 1
        source = getattr(diagram, "diagram", diagram)
        compiled = interpret(source, self.ob, dict(self.ar))
        if self._rounds is not None:
            compiled.compile(**self._rounds)
        # the diagram is pinned beside its interaction, so that a key can
        # never be a recycled ``id``.
        self._compiled[key] = (diagram, compiled)
        while len(self._compiled) > self.cache:
            self._compiled.popitem(last=False)
        return compiled

    def cache_stats(self, reset: bool = False) -> dict:
        """
        What :meth:`compile` has done so far: ``hits``, ``misses``, how
        many interactions are ``held`` and the ``capacity`` they are held
        in.  A ``held`` equal to ``capacity`` with ``misses`` still rising
        is an evicting cache, i.e. a recompilation per epoch.

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
                 "held": len(self._compiled), "capacity": self.cache}
        if reset:
            self.hits, self.misses = 0, 0
        return found

    def compile_rounds(self, **kwargs) -> MapNN:
        """
        Compile the per-round step of every interaction with
        ``torch.compile``; see :meth:`discopy.neural.CMap.compile`.
        Message passing on these maps is launch-bound -- many small kernels
        per round -- so this is a several-fold wall-clock speedup on a GPU
        at identical numerics up to rounding error.
        """
        self._rounds = kwargs
        for _, compiled in self._compiled.values():
            compiled.compile(**kwargs)
        return self

    # --- the state ----------------------------------------------------------

    def initial(self, diagram, values: Mapping = None, rows: int = None,
                dtype=None, device=None):
        """
        The initial flat state: the given values on every copy of their
        family, zeros everywhere else.

        Parameters:
            diagram : The diagram to run.
            values : A tensor of shape ``(rows, sites, width)`` per
                     ``(generator name, role)`` family.
            rows : The batch size, read off the values by default.
            dtype, device : Read off the values, or off the parameters.
        """
        interaction = self.compile(diagram)
        values = dict(values or {})
        reference = next(iter(values.values()), None)
        if reference is None:
            reference = next(iter(self.parameters()), None)
        elif rows is None:
            rows = len(reference)
        if reference is not None:
            dtype = reference.dtype if dtype is None else dtype
            device = reference.device if device is None else device
        state = interaction.zeros(1 if rows is None else rows, dtype, device)
        for key, value in values.items():
            state = interaction.write(state, key, value)
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
        return self.compile(diagram).read(state, key, every)

    def write(self, diagram, state, key, values):
        """
        A copy of the state with values written on every copy of a family.

        Parameters:
            diagram : The diagram the state belongs to.
            state : The flat messages.
            key : A ``(generator name, role)`` pair.
            values : A tensor of shape ``(rows, sites, width)``.
        """
        return self.compile(diagram).write(state, key, values)

    def sites(self, diagram, key) -> int:
        """
        How many generators of a name carry a role in a diagram.

        Parameters:
            diagram : The diagram to compile.
            key : A ``(generator name, role)`` pair.
        """
        return self.compile(diagram).sites(key)

    # --- running it ---------------------------------------------------------

    def run(self, diagram, state, deep: bool = False, **overrides):
        """
        Hand a state to the solver, returning the final state *and* the
        checkpoints -- the lower-level entry point, for a caller that
        carries its own state between calls.

        Parameters:
            diagram : The diagram to run.
            state : The flat initial messages.
            deep : Whether to keep every checkpoint.
            overrides : Test-time compute, e.g. ``rounds`` or ``steps``.
        """
        return self.solver(
            self.compile(diagram), state, deep=deep, **overrides)

    def forward(self, diagram, init=None, deep: bool = False, **overrides):
        """
        The state at the last supervised checkpoint, or the list of them
        all under ``deep``.

        Parameters:
            diagram : The diagram to run.
            init : The initial values per family, as in :meth:`initial`, or
                   a flat state to start from.
            deep : Whether to return every checkpoint.
            overrides : Test-time compute, e.g. ``rounds`` or ``steps``.
        """
        state = init if isinstance(init, torch.Tensor) \
            else self.initial(diagram, init)
        _, every = self.run(diagram, state, deep=deep, **overrides)
        return every if deep else every[-1]
