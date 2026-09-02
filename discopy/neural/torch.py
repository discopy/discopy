# -*- coding: utf-8 -*-

"""
The PyTorch backend for :mod:`discopy.neural`, imported lazily.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    PyTorch
    Zeros
    CMapModule
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from discopy.neural.backend import Backend
from discopy.neural.execution import box_forward

if TYPE_CHECKING:
    from discopy.neural.core import CMap


class PyTorch(Backend):
    """ The PyTorch neural execution backend. """

    def zeros(self, batch_size: int, width: int, like=None) -> torch.Tensor:
        """ Return a batch of zero messages. """
        if like is None:
            return torch.zeros(batch_size, width)
        return like.new_zeros((batch_size, width))

    def split(self, value: torch.Tensor, widths: tuple[int, ...]
              ) -> tuple[torch.Tensor, ...]:
        """ Split a batch into messages of the given widths. """
        return tuple(torch.split(value, widths, dim=-1))

    def concatenate(self, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """ Concatenate messages along their final dimension. """
        return torch.cat(values, dim=-1)

    def activate(self, module: torch.nn.Module, value: torch.Tensor
                 ) -> torch.Tensor:
        """ Apply a module using its nested-box protocol when available. """
        method = getattr(module, "box_forward", module)
        return method(value)

    def prototype(self, modules: tuple[torch.nn.Module, ...]):
        """ Find a parameter or buffer whose dtype and device zeros use. """
        for module in modules:
            parameter = next(module.parameters(), None)
            if parameter is not None:
                return parameter
            buffer = next(module.buffers(), None)
            if buffer is not None:
                return buffer
        return None

    def wrap(self, inside: CMap) -> CMapModule:
        """ Wrap a combinatorial map in a fresh PyTorch module. """
        return CMapModule(inside, self)

    def zeros_module(self) -> Zeros:
        """ Return a parameter-free all-port zero module. """
        return Zeros()


class Zeros(torch.nn.Module):
    """ An all-port module which emits zeros with its input's metadata. """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """ Return zeros matching the shape and metadata of ``value``. """
        return torch.zeros_like(value)


class CMapModule(torch.nn.Module):
    """
    A neural combinatorial map wrapped as a PyTorch module.

    Parameters:
        inside : The combinatorial map to wrap.
        backend : The backend executing the map.
    """
    def __init__(self, inside: CMap, backend: Backend):
        super().__init__()
        self.inside, self.backend = inside, backend
        self.networks = torch.nn.ModuleList(inside.modules)

    def forward(self, *args, **kwargs):
        """ Execute message passing over the wrapped map. """
        kwargs.update(backend=self.backend, modules=self.networks)
        return self.inside.forward(*args, **kwargs)

    def box_forward(self, messages: torch.Tensor) -> torch.Tensor:
        """ One box of the all-port protocol, see :func:`box_forward`. """
        return box_forward(
            self.inside, messages, self.backend, self.networks)
