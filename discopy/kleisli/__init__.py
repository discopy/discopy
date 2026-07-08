# -*- coding: utf-8 -*-

"""
Kleisli categories of Python monads, i.e. semantics of effectful lambda terms.

.. autosummary::
    :template: module.rst
    :toctree: ../_api

    discopy.kleisli.monad
    discopy.kleisli.channel
    discopy.kleisli.multiplicative
    discopy.kleisli.additive
"""

from discopy.kleisli import monad, channel, multiplicative, additive
from discopy.kleisli.monad import (
    Monad, Maybe, Nothing, Powerset, Subdistribution, Writer)
from discopy.kleisli.channel import Channel
