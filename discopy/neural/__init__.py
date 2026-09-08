# -*- coding: utf-8 -*-

"""
``discopy.neural`` interprets DisCoPy diagrams as neural networks.

This is the category: :class:`Dim` objects, :class:`Network` boxes and the
:class:`CMap` whose ports carry dimensions, laid out as one flat vector by
:attr:`CMap.routing`. Running a map is the business of
:mod:`discopy.neural.execution`, interpreting a diagram of another category
that of :mod:`discopy.neural.map`, and training that of
:class:`~discopy.neural.model.MapNN`; each lands on top of this module.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.core

Note
----
``import discopy.neural`` imports no tensor framework: diagrams and maps are
built, composed and rewired without one.

Example
-------
>>> assert Dim(0) == Dim() and Dim(2) @ Dim(3) == Dim(2, 3)
>>> Id(Dim(2)).transpose().to_map().boxes
()
"""

from __future__ import annotations

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
from discopy.neural import core

__all__ = [
    "CMap", "Cap", "Cup", "Diagram", "Dim", "Equation", "Functor",
    "Hypergraph", "Id", "Network", "Para", "Permutation", "Swap", "core",
]
