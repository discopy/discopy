# -*- coding: utf-8 -*-

"""
Bidirectional neural networks as string diagrams.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.network
    discopy.neural.backend
    discopy.neural.rdiff

Note that ``import discopy.neural`` does not import ``torch``: networks can be
built, composed and rewired without it, only executing them requires a
concrete :class:`discopy.neural.backend.Backend`. The default one lives in
``discopy.neural.torch``, which is left out of the summary above so that the
documentation builds without a tensor framework installed.
"""

from discopy.neural.backend import BACKENDS, Backend, backend, get_backend
from discopy.neural.network import (
    Cap,
    CMap,
    Cup,
    Diagram,
    Dim,
    Equation,
    Execution,
    Functor,
    Hypergraph,
    Id,
    Network,
    Para,
    Swap,
)
from discopy.neural import rdiff
