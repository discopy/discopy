# -*- coding: utf-8 -*-

"""
DisCoPy's Kleisli modules: monad, channel and multiplicative.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.kleisli.monad
    discopy.kleisli.channel
    discopy.kleisli.multiplicative
"""

from discopy.kleisli import monad, channel, multiplicative
from discopy.kleisli.monad import (
    Monad, Maybe, Powerset, Subdistribution, Seed)
from discopy.kleisli.channel import Channel
