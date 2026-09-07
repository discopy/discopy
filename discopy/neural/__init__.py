# -*- coding: utf-8 -*-

"""
``discopy.neural`` trains neural interpretations of DisCoPy diagrams.

:class:`~discopy.neural.model.MapNN` compiles diagram structure and shared
learnable generator maps into one :class:`CMap`, whose forward pass is the
execution formula of the geometry of interaction on any tensor framework.

The workflow
------------

A dataset of ``(diagram, inputs, target)`` samples -- the diagrams may all
differ, so long as they are built from the same generators -- a
:class:`~discopy.neural.model.MapNN` interpreting them, a
:class:`~discopy.neural.batch.Batch` for the samples whose shapes differ,
and then an ordinary PyTorch training loop::

    from discopy.neural import Dim, MapNN

    model = MapNN(
        ob={message: Dim(24), state: Dim(96)},
        ar={"cell": cell}, rounds=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for diagram, x, target in loader:
        state = model(diagram, {("cell", clue): encoder(x)})
        loss = criterion(readout(model.read(diagram, state, answer)), target)
        loss.backward(); optimizer.step(); optimizer.zero_grad()

The cells filling the generators, the solvers running the rounds and the
laws a cell promises are the notebooks' business, not the library's.

The semantics
-------------

Underneath, a diagram :math:`D` in a source category :math:`C` is
interpreted by a monoidal functor: each atomic role goes to the ``Dim`` it
carries, and each generator :math:`f : X \\to Y` to a **parametric
interaction map** on its boundary,

.. math:: \\Phi_f : \\partial f \\otimes P_f \\to \\partial f, \\qquad
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
    discopy.neural.rdiff

The framework-dependent modules -- :mod:`~discopy.neural.model` and the
:mod:`~discopy.neural.torch` and :mod:`~discopy.neural.jax` backends -- are
left out of the summary so that the documentation builds without a tensor
framework installed.


* :mod:`~discopy.neural.core` : the compact closed category itself --
  :class:`Dim` objects, :class:`Network` boxes and the :class:`CMap` whose
  forward pass is the execution formula, with the flat-state ``read`` and
  ``write`` a model addresses it through.
* :mod:`~discopy.neural.execution` : the execution formula on any
  :mod:`~discopy.neural.backend`, :mod:`torch <discopy.neural.torch>` or
  :mod:`jax <discopy.neural.jax>`: one flat array of messages, one batched
  call per group of boxes sharing a module, one permutation per round.
* :mod:`~discopy.neural.map` : the interpretation of a diagram as a map,
  the ``(generator, role)`` families of its ports, and the formal
  specifications :class:`~discopy.neural.map.ParamMap` and
  :class:`~discopy.neural.map.InteractionMap` that say what a generator
  means.
* :mod:`~discopy.neural.model` : :class:`MapNN`, the functor from diagrams
  to runnable maps as a torch module.
* :mod:`~discopy.neural.batch` : batching over heterogeneous diagrams.
* :mod:`~discopy.neural.signature` : the port layout of one generator, and
  the wiring builders that draw a diagram out of a family's combinatorics.
* :mod:`~discopy.neural.rdiff` : reverse derivatives of neural diagrams, as
  the optics of :mod:`discopy.optics`.

Note
----
``import discopy.neural`` does not import ``torch``: diagrams, signatures
and the whole compilation layer work without it.  :class:`MapNN`, the one
torch-dependent name, is imported on first use.

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
from discopy.neural import batch, core, execution, rdiff, signature
from discopy.neural.batch import Batch, bucket
from discopy.neural.map import (
    InteractionMap,
    ParamMap,
    families,
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
LAZY = ("model", )

#: The torch-dependent names, and the submodule each of them lives in.
DEFERRED = {"MapNN": "model"}

#: ``discopy.neural.map`` is a submodule, reachable as an attribute, but it
#: is deliberately kept out of ``__all__``: a star import must not shadow
#: the builtin ``map``.
__all__ = [
    "BACKENDS", "Backend", "Batch", "CMap", "Cap", "Cup", "Diagram", "Dim",
    "Equation", "Execution", "Functor", "Hypergraph", "Id", "InteractionMap",
    "MapNN", "Network", "Orbit", "Para", "ParamMap", "Permutation",
    "Signature", "Swap", "Sym", "backend", "batch", "box_ports", "bucket",
    "core", "execution", "families", "from_incidence", "from_relation",
    "from_wiring", "get_backend", "interaction_spec", "interpret", "model",
    "rdiff", "signature",
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
