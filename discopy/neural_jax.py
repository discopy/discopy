# -*- coding: utf-8 -*-

""" JAX adapter for :mod:`discopy.neural`, imported lazily. """

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from discopy.neural import Backend, CMap, ExecutionPlan


def zeros(batch_size: int, width: int, like=None):
    """ Return a batch of zero messages. """
    if like is None:
        return jnp.zeros((batch_size, width))
    return jnp.zeros_like(like, shape=(batch_size, width))


def split(value, widths: tuple[int, ...]) -> tuple:
    """ Split a batch into messages of the given widths. """
    if not widths:
        return ()
    indices = tuple(accumulate(widths[:-1]))
    return tuple(jnp.split(value, indices, axis=-1))


def concatenate(values: tuple):
    """ Concatenate messages along their final dimension. """
    return jnp.concatenate(values, axis=-1)


def activate(module, value):
    """ Apply a callable PyTree, using its nested protocol when available. """
    method = getattr(module, "box_forward", module)
    return method(value)


def prototype(modules: tuple):
    """ Find an array leaf whose dtype and placement zeros should follow. """
    for module in modules:
        for leaf in jax.tree_util.tree_leaves(module):
            if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                return leaf
    return None


def _zeros_like(value):
    return jnp.zeros_like(value)


def zeros_module():
    """ Return a parameter-free all-port zero callable PyTree. """
    return jax.tree_util.Partial(_zeros_like)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class CMapModule:
    """
    A compiled neural combinatorial map represented as a callable JAX PyTree.

    The immutable execution plan is static tree metadata, while the distinct
    modules are dynamic children visible to JAX transformations.

    Parameters:
        plan : The backend-neutral execution plan.
        modules : The callable PyTrees indexed by the plan.
    """
    plan: "ExecutionPlan"
    modules: tuple

    @property
    def backend(self):
        """ A stateless JAX backend for interpreting the execution plan. """
        from discopy.neural import JAX
        return JAX()

    def tree_flatten(self):
        """ Expose modules as dynamic children and the plan as static data. """
        return self.modules, self.plan

    @classmethod
    def tree_unflatten(cls, plan, modules):
        """ Rebuild a model after a JAX tree transformation. """
        return cls(plan, tuple(modules))

    def forward(self, x=None, init=None,
                n_rounds: int = None, inject: bool = True,
                memory=None, return_memory: bool = False,
                causal: bool = False):
        """ Execute the compiled plan with this PyTree's runtime modules. """
        from discopy.neural import Execution

        execution = Execution(
            self.plan, x, init, memory,
            backend=self.backend, modules=self.modules)
        if causal:
            if n_rounds is not None:
                raise ValueError(
                    "A causal schedule cannot be combined with n_rounds.")
            return execution.forward_causal(inject, return_memory)
        return execution.forward(n_rounds, inject, return_memory)

    __call__ = forward

    def box_forward(self, messages):
        """ Adapt direct plan execution to the neural all-port protocol. """
        from discopy.neural import Execution

        backend, plan = self.backend, self.plan
        dom_width = sum(plan.port_dims[i] for i in plan.input_ports)
        cod_width = sum(plan.port_dims[i] for i in plan.output_ports)
        memory_width = sum(plan.memory_widths)
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
            plan, memory=memory if memory_width else None,
            backend=backend, modules=self.modules)
        boundary_ports = plan.input_ports + plan.output_ports
        boundary = backend.split(
            backend.concatenate((inputs, outputs)),
            tuple(plan.port_dims[i] for i in boundary_ports))\
            if boundary_ports else ()
        initial = [None] * plan.n_ports
        for port, value in zip(boundary_ports, boundary):
            initial[plan.edges[port]] = value
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


def wrap(inside: "CMap", backend: "Backend") -> CMapModule:
    """ Wrap a combinatorial map in a fresh callable JAX PyTree. """
    del backend
    return CMapModule(inside.execution_plan, inside.modules)
