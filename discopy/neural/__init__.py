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

The modules
-----------

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.core
    discopy.neural.execution
    discopy.neural.backend
    discopy.neural.signature
    discopy.neural.map
    discopy.neural.batch

The framework-dependent modules -- :mod:`~discopy.neural.model` and the
:mod:`~discopy.neural.torch` and :mod:`~discopy.neural.jax` backends -- are
left out of the summary so that the documentation builds without a tensor
framework.

* :mod:`~discopy.neural.core` : the compact closed category itself --
  :class:`Dim` objects, :class:`Network` boxes and the :class:`CMap` whose
  forward pass is the execution formula, with the flat-state ``read`` and
  ``write`` a model addresses it through.
* :mod:`~discopy.neural.execution` : the execution formula on any
  :mod:`~discopy.neural.backend`, :mod:`torch <discopy.neural.torch>` or
  :mod:`jax <discopy.neural.jax>`: one flat array of messages, one batched
  call per group of boxes sharing a module, one permutation per round.
* :mod:`~discopy.neural.signature` : the port layout of one generator, and
  the wiring builders that draw a diagram out of a family's combinatorics.
* :mod:`~discopy.neural.map` : the interpretation of a diagram as a map,
  the ``(generator, role)`` families of its ports and their heads, and the
  width of a diagram under an interpretation; what a generator means is
  said on :class:`Network`, and a feed-forward layer is a :class:`Para`.
* :mod:`~discopy.neural.model` : :class:`MapNN`, the functor from diagrams
  to runnable maps as a torch module.
* :mod:`~discopy.neural.batch` : batching over heterogeneous diagrams.

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

from importlib import import_module

from discopy.neural.backend import BACKENDS, Backend, get_backend
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
)
from discopy.neural.execution import Execution
from discopy.neural import batch, core, execution, signature
from discopy.neural.batch import Batch, bucket
from discopy.neural.map import families, heads, interpret, to_map, width
from discopy.neural.signature import (
    Orbit,
    Signature,
    Sym,
    from_incidence,
    from_relation,
)

__all__ = [
    "BACKENDS", "Backend", "Batch", "CMap", "Cap", "Cup", "Diagram", "Dim",
    "Equation", "Execution", "Functor", "Hypergraph", "Id", "Network", "Orbit",
    "Para", "Permutation", "Signature", "Swap", "Sym", "batch", "bucket",
    "core", "execution", "families", "from_incidence", "from_relation",
    "get_backend", "heads", "interpret", "signature", "to_map", "width",
]
"""
``discopy.neural.map`` is a submodule, reachable as an attribute, but it
is deliberately kept out of ``__all__``: a star import must not shadow
the builtin ``map``. ``model`` and ``MapNN`` import torch, so they are
imported on first use and kept out of a star import too.
"""


def __getattr__(name: str):
    """ Import the torch-dependent ``model`` or ``MapNN`` on first use. """
    if name not in ("model", "MapNN"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = import_module("discopy.neural.model")
    return model if name == "model" else model.MapNN
