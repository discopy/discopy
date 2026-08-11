# -*- coding: utf-8 -*-

"""
The compact closed category of bidirectional neural networks, and the
category-generic engine that runs message passing over it.

:mod:`discopy.neural.core` is the category itself: :class:`Dim` objects,
:class:`Network` boxes and the :class:`CMap` whose forward pass is the
execution formula of the geometry of interaction.  It is re-exported here,
so ``from discopy.neural import Dim, Network, CMap`` keeps working.

The target category never changes -- ``Dim`` is self-dual, routing is a
permutation of a flat tensor, a trace is a self-wired pair of ports -- and
that is what makes one engine enough.  Swaps, cups, caps and traces become
wiring, which a functor preserves strictly and for free; a box whose legs
carry a symmetry stays a box, and its equations hold iff its torch module
satisfies them.  Supporting a new source category is therefore not new
wiring code but a declaration: :mod:`discopy.neural.signature` says which
equations a box must satisfy, :mod:`discopy.neural.cells` generates a
module that satisfies them, and
:func:`~discopy.neural.signature.check_equivariant` measures whether it
does.

* :mod:`~discopy.neural.core` : the category, moved verbatim.
* :mod:`~discopy.neural.signature` : :class:`~signature.Sym`,
  :class:`~signature.Orbit`, :class:`~signature.Signature` and
  :func:`~signature.check_equivariant` -- the port layout of a box, stated
  once.
* :mod:`~discopy.neural.skeleton` : abstract wirings, built from a task's
  combinatorics in any source category.
* :mod:`~discopy.neural.functor` : the interpretation of a skeleton into a
  runnable map, and the :class:`~functor.Router` that reads its ports.
* :mod:`~discopy.neural.cells` : the two module shapes a site can carry --
  a stateful node and a stateless hyperedge.
* :mod:`~discopy.neural.engine` : the message-passing schedules, from one
  supervised run to segmented recursion with adaptive computation time.
* :mod:`~discopy.neural.batch` : batching over variable structure as the
  monoidal product of maps.

and three modules that say what the others *mean*, without computing
anything:

* :mod:`~discopy.neural.parametric` : a box as a parametric interaction
  map :math:`\\Phi : P \\otimes (X^* \\otimes Y) \\to X^* \\otimes Y` on its
  boundary, rather than as a feed-forward :math:`X \\to Y`.
* :mod:`~discopy.neural.dynamics` : a map as a global transition
  :math:`T_\\theta = \\sigma_D \\circ \\Phi_\\theta`, and running it as finite
  iteration.
* :mod:`~discopy.neural.laws` : a signature's symmetry as a group action,
  and how strongly a learned module keeps it.

Note
----
``import discopy.neural`` does not import ``torch``: networks can be
built, composed and rewired without it, and so can signatures, skeletons
and the whole formal layer.  The three modules that *are* torch --
``cells``, ``engine`` and ``batch`` -- are imported on first use.

Example
-------
>>> assert Dim(0) == Dim() and Dim(2) @ Dim(3) == Dim(2, 3)
>>> Id(Dim(2)).transpose().to_map().boxes
()
"""

from __future__ import annotations

import importlib

from discopy.neural.core import (
    CMap,
    Cap,
    Cup,
    Diagram,
    Dim,
    Functor,
    Hypergraph,
    Id,
    Network,
    Swap,
)
from discopy.neural.core import Box
from discopy.neural import (
    core, dynamics, functor, laws, parametric, signature, skeleton)
from discopy.neural.dynamics import Iteration, Transition
from discopy.neural.functor import Interpretation, Router, Wiring, interpret
from discopy.neural.laws import Action, Law, Strictness
from discopy.neural.parametric import (
    InteractionMap,
    ParamMap,
    Parametric,
    interaction_spec,
)
from discopy.neural.signature import (
    Orbit,
    Signature,
    Sym,
    check_equivariant,
)
from discopy.neural.skeleton import Skeleton

#: The submodules that import ``torch`` at module level, loaded lazily so
#: that ``import discopy.neural`` stays torch-free.
LAZY = ("batch", "cells", "engine")

__all__ = [
    "Action", "Box", "CMap", "Cap", "Cup", "Diagram", "Dim", "Functor",
    "Hypergraph", "Id", "InteractionMap", "Interpretation", "Iteration",
    "Law", "Network", "Orbit", "ParamMap", "Parametric", "Router",
    "Signature", "Skeleton", "Strictness", "Swap", "Sym", "Transition",
    "Wiring", "batch", "cells", "check_equivariant", "core", "dynamics",
    "engine", "functor", "interaction_spec", "interpret", "laws",
    "parametric", "signature", "skeleton",
]


def __getattr__(name: str):
    """ Import a torch-dependent submodule on first use. """
    if name in LAZY:
        module = importlib.import_module(f"discopy.neural.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
