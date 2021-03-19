# -*- coding: utf-8 -*-
# pylint: disable=unused-import

"""
Groups together the grammar modules:
:mod:`pregroup`, :mod:`cfg` and :mod:`ccg`
"""

from discopy.grammar import pregroup, cfg, ccg
from discopy.grammar.pregroup import (
    Word, draw, eager_parse, brute_force, normal_form)
from discopy.grammar.cfg import CFG
from discopy.grammar.ccg import cat2ty, tree2diagram
