# -*- coding: utf-8 -*-

"""
The abstract interface a neural execution backend has to implement.

A backend owns the tensor primitives and the module protocol, so that
:mod:`discopy.neural.network` only knows about the geometry of interaction.
Concrete backends live in their own module, e.g. :mod:`discopy.neural.torch`,
and are imported lazily so that ``import discopy.neural`` imports no tensor
framework.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Backend

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        backend
        get_backend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module


class Backend(ABC):
    """ An abstract neural execution backend. """

    @abstractmethod
    def zeros(self, batch_size: int, width: int, like=None):
        """ Return a batch of zero messages. """

    @abstractmethod
    def split(self, value, widths: tuple[int, ...]) -> tuple:
        """ Split a batch into messages of the given widths. """

    @abstractmethod
    def concatenate(self, values: tuple):
        """ Concatenate messages along their final dimension. """

    @abstractmethod
    def activate(self, module, value):
        """ Apply a backend-owned module to an all-port message. """

    @abstractmethod
    def prototype(self, modules: tuple):
        """ Find a value whose dtype and device zero messages should use. """

    @abstractmethod
    def wrap(self, inside):
        """ Wrap a combinatorial map as a backend-owned module. """

    @abstractmethod
    def zeros_module(self):
        """ Return a parameter-free all-port zero module. """


BACKENDS = {
    'pytorch': 'discopy.neural.torch.PyTorch',
    'jax': 'discopy.neural.jax.JAX',
}

_current = ContextVar('discopy.neural.backend', default='pytorch')
_cache = {}


@contextmanager
def backend(name: str = None):
    """
    Context manager for neural execution backends.

    The backend classes of :data:`BACKENDS` are given by qualified name and
    imported when they are first used, so that building and rewiring networks
    needs no tensor framework. The current backend is stored in a
    :class:`~contextvars.ContextVar` so that concurrent threads or tasks do
    not share each other's selection.

    Parameters:
        name : The backend name, ``"pytorch"`` by default.
    """
    name = name or _current.get()
    token = _current.set(name)
    try:
        if name not in _cache:
            module, _, cls = BACKENDS[name].rpartition('.')
            _cache[name] = getattr(import_module(module), cls)()
        yield _cache[name]
    finally:
        _current.reset(token)


def current() -> str:
    """ The name of the backend selected by the innermost :func:`backend`. """
    return _current.get()


def get_backend(name: str | Backend = None) -> Backend:
    """
    Get a neural execution backend by name, or return a given backend.

    Parameters:
        name : The backend name or instance, the current backend by default.
    """
    if isinstance(name, Backend):
        return name
    with backend(name) as result:
        return result
