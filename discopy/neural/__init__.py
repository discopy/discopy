# -*- coding: utf-8 -*-

"""
``discopy.neural`` interprets DisCoPy diagrams as neural networks.

:mod:`~discopy.neural.core` is the category: :class:`Dim` objects,
:class:`Network` boxes and the :class:`CMap` whose ports carry dimensions,
laid out as one flat vector by :attr:`CMap.routing`.
:mod:`~discopy.neural.execution` runs a map on a
:mod:`~discopy.neural.backend`: :meth:`CMap.forward` is the execution
formula of the geometry of interaction, all the messages in one flat array,
one batched call per group of boxes sharing a module, one permutation per
round. :mod:`~discopy.neural.signature` says what a generator of the source
category promises: its ports grouped into orbits, the symmetry each orbit
carries, and the wiring builders that draw a diagram out of a family's
combinatorics. Interpreting such a diagram is the business of
:mod:`discopy.neural.map` and training that of
:class:`~discopy.neural.model.MapNN`; both land on top of these.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.core
    discopy.neural.execution
    discopy.neural.backend
    discopy.neural.signature

The framework-dependent backends :mod:`~discopy.neural.torch` and
:mod:`~discopy.neural.jax` are left out of the summary so that the
documentation builds without a tensor framework.

Note
----
``import discopy.neural`` imports no tensor framework: diagrams and maps are
built, composed and rewired without one, only running them needs a backend.

Example
-------
>>> assert Dim(0) == Dim() and Dim(2) @ Dim(3) == Dim(2, 3)
>>> Id(Dim(2)).transpose().to_map().boxes
()
"""

from __future__ import annotations

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
from discopy.neural import core, execution, signature
from discopy.neural.signature import (
    Orbit,
    Signature,
    Sym,
    from_incidence,
    from_relation,
)

__all__ = [
    "BACKENDS", "Backend", "CMap", "Cap", "Cup", "Diagram", "Dim", "Equation",
    "Execution", "Functor", "Hypergraph", "Id", "Network", "Orbit", "Para",
    "Permutation", "Signature", "Swap", "Sym", "core", "execution",
    "from_incidence", "from_relation", "get_backend", "signature",
]
