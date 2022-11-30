# -*- coding: utf-8 -*-
# pylint: disable=unused-import

"""
DisCoPy's grammar modules: categorial and pregroup.


Example
-------
>>> s, n, v, vp = Ty('S'), Ty('N'), Ty('V'), Ty('VP')
>>> R0, R1 = Box('R0', vp @ n, s), Box('R1', n @ v , vp)
>>> Jane, loves = Word('Jane', n), Word('loves', v)
>>> sentence = Jane >> loves @ Id(N) >> Jane @ Id(V @ N) >> R1 @ Id(N) >> R0
>>> sentence.draw(figsize=(4, 3),\\
... path='docs/imgs/grammar/cfg-example.png')

.. image:: /imgs/grammar/cfg-example.png
    :align: center
"""

from discopy.grammar import pregroup, categorial
from discopy.grammar.pregroup import (
    Word, draw, eager_parse, brute_force, normal_form)
from discopy.grammar.categorial import cat2ty, tree2diagram

from discopy import messages
from discopy.monoidal import Ty, Box, Id
