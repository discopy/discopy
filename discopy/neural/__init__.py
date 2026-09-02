# -*- coding: utf-8 -*-

"""
``discopy.neural`` trains neural interpretations of DisCoPy diagrams.

:class:`~discopy.neural.model.MapNN` compiles diagram structure and shared
learnable generator maps into a global interaction, while a
:class:`~discopy.neural.solver.Solver` specifies how that interaction is
executed.

The workflow
------------

A dataset of ``(diagram, inputs, target)`` samples -- the diagrams may all
differ, so long as they are built from the same generators -- a
:class:`~discopy.neural.model.MapNN` interpreting them, a
:class:`~discopy.neural.batch.Batch` for the samples whose shapes differ, a
solver, and then an ordinary PyTorch training loop::

    from discopy.neural import Dim, Iterate, MapNN, Mode, Site

    model = MapNN(
        ob={message: Dim(24), state: Dim(96)},
        ar={"cell": Site(cell, widths, {state: Mode.STATE}, hidden=192)},
        solver=Iterate(rounds=16))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for diagram, x, target in loader:
        state = model(diagram, {("cell", clue): encoder(x)})
        loss = criterion(readout(model.read(diagram, state, answer)), target)
        loss.backward(); optimizer.step(); optimizer.zero_grad()

``docs/neural/examples/sudoku`` is that workflow at full size.

The semantics
-------------

Underneath, a diagram :math:`D` in a source category :math:`C` is
interpreted by a monoidal functor: each atomic role goes to the ``Dim`` it
carries, and each generator :math:`f : X \\to Y` to a **parametric
interaction map** on its boundary,

.. math:: \\Phi_f : P_f \\otimes \\partial f \\to \\partial f, \\qquad
          \\partial f = X^* \\otimes Y,

rather than to an ordinary feed-forward map :math:`X \\to Y`.  Wiring the
boundaries together compiles the diagram to a global transition

.. math:: T_{D,\\theta} = \\sigma_D \\circ \\Phi_\\theta : S_D \\to S_D,

the execution formula of the geometry of interaction, on the state object
:math:`S_D` with one summand per port.  Swaps, cups, caps and traces are
wiring in the target category, so a functor preserves them strictly and for
free; what survives as a box is a generator whose legs carry a symmetry,
and *that* is a promise about a torch module, measured rather than
assumed.

The modules
-----------

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.core
    discopy.neural.execution
    discopy.neural.backend
    discopy.neural.map
    discopy.neural.batch
    discopy.neural.signature
    discopy.neural.laws
    discopy.neural.rdiff

The torch-dependent modules -- :mod:`~discopy.neural.model`,
:mod:`~discopy.neural.solver`, :mod:`~discopy.neural.cells` and the
:mod:`~discopy.neural.torch` and :mod:`~discopy.neural.jax` backends -- are
left out of the summary so that the documentation builds without a tensor
framework installed.


* :mod:`~discopy.neural.model` : :class:`MapNN`, the central abstraction.
* :mod:`~discopy.neural.map` : the interpretation and the compiled
  :class:`~discopy.neural.map.Interaction`, together with the formal
  specifications :class:`~discopy.neural.map.ParamMap` and
  :class:`~discopy.neural.map.InteractionMap` that say what a generator
  means.
* :mod:`~discopy.neural.solver` : :class:`~discopy.neural.solver.Iterate`,
  :class:`~discopy.neural.solver.FixedPoint`,
  :class:`~discopy.neural.solver.Recursion` and
  :class:`~discopy.neural.solver.ACT`.
* :mod:`~discopy.neural.batch` : batching over heterogeneous diagrams.
* :mod:`~discopy.neural.cells` : the concrete neural interpretations a
  generator can carry.
* :mod:`~discopy.neural.signature` : the port layout of one generator, and
  the wiring builders that draw a diagram out of a family's combinatorics.
* :mod:`~discopy.neural.laws` : the equations a generator promises, and how
  strongly a learned module keeps them.
* :mod:`~discopy.neural.core` : the compact closed category itself --
  :class:`Dim` objects, :class:`Network` boxes and the :class:`CMap` whose
  forward pass is the execution formula, vectorised on torch.
* :mod:`~discopy.neural.execution` : the execution formula one call per box
  per round, on any :mod:`~discopy.neural.backend`: :mod:`torch
  <discopy.neural.torch>` and :mod:`jax <discopy.neural.jax>`.
* :mod:`~discopy.neural.rdiff` : reverse derivatives of neural diagrams, as
  the optics of :mod:`discopy.optics`.

Note
----
``import discopy.neural`` does not import ``torch``: diagrams, signatures,
laws and the whole compilation layer work without it.  The torch-dependent
names -- :class:`MapNN`, the solvers and the cells -- are imported on first
use.

Example
-------
>>> assert Dim(0) == Dim() and Dim(2) @ Dim(3) == Dim(2, 3)
>>> Id(Dim(2)).transpose().to_map().boxes
()
"""

from __future__ import annotations

import importlib

from discopy.neural.backend import BACKENDS, Backend, backend, get_backend
from discopy.neural.core import (
    CMap,
    Cap,
    Cup,
    Diagram,
    Dim,
    Equation,
    Functor,
    Hypergraph,
    Id,
    Network,
    Para,
    Permutation,
    Swap,
    box_ports,
    from_wiring,
)
from discopy.neural.execution import Execution
from discopy.neural import batch, core, execution, laws, rdiff, signature
from discopy.neural.batch import Batch, bucket
from discopy.neural.laws import (
    Action,
    Law,
    Strictness,
    check_equivariant,
    fusion_residual,
    symmetry,
)
from discopy.neural.map import (
    Interaction,
    InteractionMap,
    ParamMap,
    interaction_spec,
    interpret,
)
from discopy.neural.signature import (
    Orbit,
    Signature,
    Sym,
    from_incidence,
    from_relation,
)

#: The submodules that import ``torch`` at module level, loaded lazily so
#: that ``import discopy.neural`` stays torch-free.
LAZY = ("cells", "model", "solver")

#: The torch-dependent names, and the submodule each of them lives in.
DEFERRED = {
    "MapNN": "model",
    "ACT": "solver", "FixedPoint": "solver", "HaltHead": "solver",
    "Iterate": "solver", "Recursion": "solver", "Refresh": "solver",
    "Solver": "solver",
    "Cyclic": "cells", "Gate": "cells", "Mode": "cells",
    "Relation": "cells", "Site": "cells",
}

#: ``discopy.neural.map`` is a submodule, reachable as an attribute, but it
#: is deliberately kept out of ``__all__``: a star import must not shadow
#: the builtin ``map``.
__all__ = [
    "ACT", "Action", "BACKENDS", "Backend", "Batch", "CMap", "Cap", "Cup",
    "Cyclic", "Diagram", "Dim", "Equation", "Execution", "FixedPoint",
    "Functor", "Gate", "HaltHead", "Hypergraph", "Id", "Interaction",
    "InteractionMap", "Iterate", "Law", "MapNN", "Mode", "Network", "Orbit",
    "Para", "ParamMap", "Permutation", "Recursion", "Refresh", "Relation",
    "Signature", "Site", "Solver", "Strictness", "Swap", "Sym", "backend",
    "batch", "box_ports", "bucket", "cells", "check_equivariant", "core",
    "execution", "from_incidence", "from_relation", "from_wiring",
    "fusion_residual", "get_backend", "interaction_spec", "interpret",
    "laws", "model", "rdiff", "signature", "solver", "symmetry",
]


def __getattr__(name: str):
    """ Import a torch-dependent submodule or name on first use. """
    if name in LAZY:
        module = importlib.import_module(f"discopy.neural.{name}")
        globals()[name] = module
        return module
    if name in DEFERRED:
        module = importlib.import_module(f"discopy.neural.{DEFERRED[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
