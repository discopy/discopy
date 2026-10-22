# -*- coding: utf-8 -*-

"""
DisCopy's grammar modules:
thue, cfg, categorial, pregroup, dependency, abstract, proofnet and neural,
the last one needing PyTorch, which the ``grammar`` extra installs, and
imported on its own.
"""

from discopy.grammar import (
    thue, cfg, categorial, pregroup, dependency, abstract, proofnet)
from discopy.grammar.thue import Word, Rule
