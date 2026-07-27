# -*- coding: utf-8 -*-

""" JAX adapter for :mod:`discopy.neural`, imported lazily. """

from __future__ import annotations

from itertools import accumulate

import jax
import jax.numpy as jnp


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
