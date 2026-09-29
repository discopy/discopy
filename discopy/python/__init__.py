# -*- coding: utf-8 -*-

"""
Categories of Python functions.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    finset
    function
    additive
    multiplicative
"""

from importlib import import_module


def __getattr__(name):
    """
    The functions are imported on first use rather than with the package:
    :mod:`multiplicative` imports :mod:`monoidal`, which imports :mod:`finset`.
    """
    if name not in ("exp", "Ty", "Function"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("discopy.python.multiplicative"), name)
