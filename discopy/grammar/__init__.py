# -*- coding: utf-8 -*-
# pylint: disable=unused-import

"""
DisCoPy's grammar modules: formal, categorial and pregroup.
"""

from discopy.grammar import pregroup, categorial
from discopy.grammar.pregroup import (
    Word, draw, eager_parse, brute_force, normal_form)
from discopy.grammar.categorial import cat2ty, tree2diagram

from discopy import messages
from discopy.monoidal import Ty, Box, Id
