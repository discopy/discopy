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
import numpy

from discopy.neural.backend import Backend
from discopy.neural.execution import box_forward

if TYPE_CHECKING:
    from discopy.neural.core import CMap


class JAX(Backend):
    """
    The JAX neural execution backend.

    A module is a callable PyTree from one batched all-port array to an array
    of the same width. ``jax.tree_util.Partial`` can bind parameter arrays to
    a function while leaving them visible to JAX transformations.
    :meth:`CMap.as_network <discopy.neural.core.CMap.as_network>` returns a
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

    def index(self, indices: tuple[int, ...], like=None):
        """ Return an integer array of positions, concrete under ``jit``. """
        return numpy.asarray(indices, dtype=numpy.int32)

    def put(self, value, indices, updates):
        """ Return a copy of ``value`` with ``updates`` at ``indices``. """
        return value.at[:, indices].set(updates)

    def compile(self, function, **kwargs):
        """
        Return the function under ``jax.jit``, with ``inject`` static: the
        round step of :func:`~discopy.neural.execution.make_step` branches
        on it.
        """
        return jax.jit(function, **{"static_argnames": ("inject", ), **kwargs})


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
        """ One box of the all-port protocol, see :func:`box_forward`. """
        return box_forward(
            self.inside, messages, self.backend, self.modules)
