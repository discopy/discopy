# -*- coding: utf-8 -*-
# pylint: disable=unused-import

"""
Groups together the grammar modules:
:mod:`pregroup`, :mod:`cfg` and :mod:`ccg`
"""

from discopy import pregroup, cfg, ccg
from discopy.pregroup import Word, draw, eager_parse, brute_force
from discopy.cfg import CFG
from discopy.ccg import cat2ty, tree2diagram
