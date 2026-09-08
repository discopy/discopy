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

from discopy.python.multiplicative import exp, Function


def __getattr__(name):
    if name == "Ty":
        from discopy.python.multiplicative import Ty
        return Ty
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
