# -*- coding: utf-8 -*-

"""
The JAX backend for :mod:`discopy.neural`, imported lazily.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    JAX
    CMapModule
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from discopy.neural.backend import Backend

if TYPE_CHECKING:
    from discopy.neural.network import CMap


class JAX(Backend):
    """
    The JAX neural execution backend.

    A module is a callable PyTree from one batched all-port array to an array
    of the same width. ``jax.tree_util.Partial`` can bind parameter arrays to
    a function while leaving them visible to JAX transformations.
    :meth:`CMap.as_network <discopy.neural.network.CMap.as_network>` returns a
    callable PyTree whose map is static metadata and whose distinct modules are
    dynamic children. Pass that wrapper as an argument to transformations, e.g.
    ``jax.jit(lambda model, x: model(x))(model, x)``. Execution controls such
    as ``n_rounds``, ``inject``, ``causal`` and ``return_memory`` stay static
    Python values under JIT. Passing the wrapper itself directly to
    ``jax.jit`` closes over its current parameters; pass it as an argument
    when they should remain dynamic.
    """

    def zeros(self, batch_size: int, width: int, like=None):
        """ Return a batch of zero messages. """
        if like is None:
            return jnp.zeros((batch_size, width))
        return jnp.zeros_like(like, shape=(batch_size, width))

    def split(self, value, widths: tuple[int, ...]) -> tuple:
        """ Split a batch into messages of the given widths. """
        if not widths:
            return ()
        return tuple(jnp.split(value, tuple(accumulate(widths[:-1])), axis=-1))

    def concatenate(self, values: tuple):
        """ Concatenate messages along their final dimension. """
        return jnp.concatenate(values, axis=-1)

    def activate(self, module, value):
        """ Apply a PyTree using its nested-box protocol when available. """
        method = getattr(module, "box_forward", module)
        return method(value)

    def prototype(self, modules: tuple):
        """ Find an array leaf whose dtype and placement zeros follow. """
        for module in modules:
            for leaf in jax.tree_util.tree_leaves(module):
                if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                    return leaf
        return None

    def wrap(self, inside: CMap) -> CMapModule:
        """ Wrap a combinatorial map in a fresh callable PyTree. """
        return CMapModule(inside, tuple(inside.modules), self)

    def zeros_module(self):
        """ Return a parameter-free all-port zero callable PyTree. """
        return jax.tree_util.Partial(jnp.zeros_like)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class CMapModule:
    """
    A neural combinatorial map wrapped as a callable JAX PyTree.

    The map and the stateless backend are static tree metadata, while the
    distinct modules are dynamic children visible to JAX transformations.

    Parameters:
        inside : The combinatorial map to wrap.
        modules : The callable PyTrees indexed by the map.
        backend : The stateless backend executing the map.
    """
    inside: CMap
    modules: tuple
    backend: Backend

    def tree_flatten(self):
        """ Expose the modules as children and the map as static data. """
        return self.modules, (self.inside, self.backend)

    @classmethod
    def tree_unflatten(cls, metadata, modules):
        """ Rebuild a wrapper after a JAX tree transformation. """
        inside, backend = metadata
        return cls(inside, tuple(modules), backend)

    def forward(self, *args, **kwargs):
        """ Execute message passing over the wrapped map. """
        kwargs.update(backend=self.backend, modules=self.modules)
        return self.inside.forward(*args, **kwargs)

    __call__ = forward

    def box_forward(self, messages):
        """ Adapt direct map execution to the neural all-port protocol. """
        from discopy.neural.network import Execution

        backend, inside = self.backend, self.inside
        dom_width, cod_width = sum(inside.dom.inside), sum(inside.cod.inside)
        memory_width = sum(sum(box.mem.inside) for box in inside.boxes)
        expected = dom_width + cod_width + memory_width
        shape = getattr(messages, "shape", None)
        if shape is None or len(shape) != 2 or shape[-1] != expected:
            actual = None if shape is None else tuple(shape)
            raise ValueError(
                f"Nested map messages have shape {actual}, "
                f"expected (batch_size, {expected}).")
        inputs, outputs, memory = backend.split(
            messages, (dom_width, cod_width, memory_width))
        execution = Execution(
            inside, memory=memory if memory_width else None,
            backend=backend, modules=self.modules)
        boundary_ports = inside.input_ports + inside.output_ports
        boundary = backend.split(
            backend.concatenate((inputs, outputs)),
            tuple(inside.port_dims[i] for i in boundary_ports))\
            if boundary_ports else ()
        initial = [None] * inside.n_ports
        for port, value in zip(boundary_ports, boundary):
            initial[inside.edges[port]] = value
        execution.init = initial
        execution.forward()
        public = backend.concatenate(tuple(
            execution.incoming[i] for i in boundary_ports))\
            if boundary_ports\
            else backend.zeros(messages.shape[0], 0, like=messages)
        next_memory = backend.concatenate(execution.memories)\
            if execution.memories\
            else backend.zeros(messages.shape[0], 0, like=messages)
        return backend.concatenate((public, next_memory))
