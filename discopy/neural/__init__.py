# -*- coding: utf-8 -*-

"""
``discopy.neural`` interprets DisCoPy diagrams as neural networks:
:mod:`discopy.neural.network` is the traced category of feedforward
networks, with lists of shapes :class:`Dims` as objects and layers as boxes.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.neural.network

Note
----
``import discopy.neural`` imports no tensor framework: networks are built,
composed and rewired without one.
"""

from discopy.neural import network
from discopy.neural.network import Dim, Dims, Network

__all__ = ["Dim", "Dims", "Network", "network"]
