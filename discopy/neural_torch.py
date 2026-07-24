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


def activate(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """ Apply a module using its nested-box protocol when available. """
    method = getattr(module, "box_forward", module)
    return method(value)


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

    def box_forward(self, messages: torch.Tensor) -> torch.Tensor:
        """ Adapt direct map execution to the neural all-port protocol. """
        from discopy.neural import Execution

        dom_width, cod_width = (
            sum(self.inside.dom.inside), sum(self.inside.cod.inside))
        memory_width = sum(
            sum(box.mem.inside) for box in self.inside.boxes)
        expected = dom_width + cod_width + memory_width
        if len(messages.shape) != 2 or messages.shape[-1] != expected:
            raise ValueError(
                f"Nested map messages have shape {tuple(messages.shape)}, "
                f"expected (batch_size, {expected}).")
        inputs, outputs, memory = messages.split(
            (dom_width, cod_width, memory_width), dim=-1)
        execution = Execution(
            self.inside, memory=memory if memory_width else None)
        boundary_ports = execution.input_ports + execution.output_ports
        boundary = torch.split(
            torch.cat((inputs, outputs), dim=-1),
            tuple(self.inside.port_dims[i] for i in boundary_ports), dim=-1)\
            if boundary_ports else ()
        initial = [None] * self.inside.n_ports
        for port, value in zip(boundary_ports, boundary):
            initial[self.inside.edges[port]] = value
        execution.init = initial
        execution.forward()
        public = torch.cat(
            tuple(execution.incoming[i] for i in boundary_ports), dim=-1)\
            if boundary_ports\
            else messages.new_zeros((messages.shape[0], 0))
        next_memory = torch.cat(execution.memories, dim=-1)\
            if execution.memories\
            else messages.new_zeros((messages.shape[0], 0))
        return torch.cat((public, next_memory), dim=-1)


def wrap(inside: "CMap") -> CMapModule:
    """ Wrap a combinatorial map in a fresh PyTorch module. """
    return CMapModule(inside)
