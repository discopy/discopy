# -*- coding: utf-8 -*-

"""
DisCoPy's Kleisli modules: monad, channel, additive, multiplicative and
token.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.kleisli.monad
    discopy.kleisli.channel
    discopy.kleisli.additive
    discopy.kleisli.multiplicative
    discopy.kleisli.token
"""

from discopy.kleisli import (
    monad, channel, additive, multiplicative, token)
from discopy.kleisli.monad import (
    Monad, Maybe, Powerset, Subdistribution, Seed)
from discopy.kleisli.channel import Channel
from discopy.kleisli.token import Machine
