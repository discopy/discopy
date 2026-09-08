# -*- coding: utf-8 -*-

"""
``discopy.neural`` interprets DisCoPy diagrams as neural networks.

:mod:`discopy.neural.network` is the traced category of feedforward
networks, with lists of shapes :class:`Dims` as objects and layers as boxes,
and :mod:`discopy.neural.interaction` is the free compact category it
generates, whose combinatorial maps :class:`CMap` run networks against each
other along their legs.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.network
    discopy.neural.interaction

Note
----
``import discopy.neural`` imports no tensor framework: networks are built,
composed and rewired without one.
"""

from discopy.neural import network, interaction
from discopy.neural.interaction import CMap
from discopy.neural.network import Dim, Dims, Network

__all__ = ["CMap", "Dim", "Dims", "Network", "interaction", "network"]
