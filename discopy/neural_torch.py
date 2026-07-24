# -*- coding: utf-8 -*-

""" PyTorch adapter for :mod:`discopy.neural`, imported lazily. """

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from discopy.neural import CMap


def zeros(batch_size: int, width: int, like=None) -> torch.Tensor:
    """ Return a batch of zero messages. """
    if like is None:
        return torch.zeros(batch_size, width)
    return like.new_zeros((batch_size, width))


def split(
        value: torch.Tensor, widths: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """ Split a batch into messages of the given widths. """
    return tuple(torch.split(value, widths, dim=-1))


def concatenate(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """ Concatenate messages along their final dimension. """
    return torch.cat(values, dim=-1)


def prototype(modules: tuple[torch.nn.Module, ...]):
    """ Find a parameter or buffer whose dtype and device zeros should use. """
    for module in modules:
        parameter = next(module.parameters(), None)
        if parameter is not None:
            return parameter
        buffer = next(module.buffers(), None)
        if buffer is not None:
            return buffer
    return None


class Zeros(torch.nn.Module):
    """ An all-port module which emits zeros with its input's metadata. """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """ Return zeros matching the shape and metadata of ``value``. """
        return torch.zeros_like(value)


def zeros_module() -> Zeros:
    """ Return a parameter-free all-port zero module. """
    return Zeros()


class CMapModule(torch.nn.Module):
    """
    A neural combinatorial map wrapped as a PyTorch module.

    Parameters:
        inside : The combinatorial map to wrap.
    """
    def __init__(self, inside: "CMap"):
        super().__init__()
        self.inside = inside
        self.networks = torch.nn.ModuleList(inside.modules)

    def forward(self, *args, **kwargs):
        """ Execute message passing over the wrapped map. """
        return self.inside.forward(*args, **kwargs)


def wrap(inside: "CMap") -> CMapModule:
    """ Wrap a combinatorial map in a fresh PyTorch module. """
    return CMapModule(inside)
