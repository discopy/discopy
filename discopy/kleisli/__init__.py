# -*- coding: utf-8 -*-

"""
DisCoPy's Kleisli modules: monad, channel, additive and multiplicative.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.kleisli.monad
    discopy.kleisli.channel
    discopy.kleisli.additive
    discopy.kleisli.multiplicative
"""

from discopy.kleisli import monad, channel, additive, multiplicative
from discopy.kleisli.monad import (
    Monad, Maybe, Powerset, Subdistribution, Seed)
from discopy.kleisli.channel import Channel
