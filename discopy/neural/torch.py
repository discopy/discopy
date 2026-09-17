# -*- coding: utf-8 -*-

"""
A feedforward network as a PyTorch module.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Module

Example
-------
>>> import torch  # doctest: +EXTRA
>>> from discopy.neural.network import Box, Dims, Network
>>> x, h = Dims(4), Dims(8)
>>> layer = Box('layer', x, h, module=torch.nn.Linear(4, 8))
>>> relu = Box('relu', h, h, module=torch.nn.ReLU())
>>> head = Box('head', h, x, module=torch.nn.Linear(8, 4))
>>> add = Box('add', x @ x, x, module=torch.add)
>>> residual = Network.copy(x) >> (layer >> relu >> head) @ x >> add
>>> module = Module(residual)
>>> module(torch.ones(5, 4)).shape
torch.Size([5, 4])
>>> len(list(module.parameters()))
4
"""

from __future__ import annotations

import torch

from discopy.neural.network import Network


class Module(torch.nn.Module):
    """
    A network as a PyTorch module: the modules of its boxes are its
    submodules, so that their parameters are its parameters, and ``forward``
    runs the network on one tensor per input leg, returning one per output
    leg.

    Parameters:
        network : The feedforward network to run.

    Note
    ----
    A module shared by several boxes is registered once, so that the boxes
    share its weights. ``torch.compile`` takes the module like any other.
    """
    def __init__(self, network: Network):
        super().__init__()
        self.network = network
        self.boxes = torch.nn.ModuleList(dict.fromkeys(
            box.module for box in network.boxes
            if isinstance(box.module, torch.nn.Module)))
        self.function = network.to_function(torch.Tensor)

    def forward(self, *tensors: torch.Tensor):
        return self.function(*tensors)
