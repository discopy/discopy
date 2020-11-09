# -*- coding: utf-8 -*-

"""
Implements combinatory categorial grammars.
"""

import re

from discopy.grammar import cfg
from discopy.biclosed import Ty, Box, Id, FA, BA, FC


class Word(cfg.Word, Box):
    """ Word with a :class:`discopy.biclosed.Ty` as codomain. """


def cat2ty(string):
    """
    Takes the string repr of a CCG category,
    returns a :class:`discopy.biclosed.Ty`. """
    def unbracket(string):
        return string[1:-1] if string[0] == '(' else string

    def remove_modifier(string):
        return re.sub(r'\[[^]]*\]', '', string)

    def split(string):
        par_count = 0
        for i, char in enumerate(string):
            if char == "(":
                par_count += 1
            elif char == ")":
                par_count -= 1
            elif char in ["\\", "/"] and par_count == 0:
                return unbracket(string[:i]), char, unbracket(string[i + 1:])
        return remove_modifier(string), None, None

    left, slash, right = split(string)
    if slash == '\\':
        return cat2ty(right) >> cat2ty(left)
    if slash == '/':
        return cat2ty(left) << cat2ty(right)
    return Ty(left)


def tree2diagram(tree, dom=Ty()):
    """
    Takes a depccg.Tree in JSON format,
    returns a :class:`discopy.biclosed.Diagram`.
    """
    if 'word' in tree:
        return Word(tree['word'], cat2ty(tree['cat']), dom=dom)
    children = list(map(tree2diagram, tree['children']))
    dom = Ty().tensor(*[child.cod for child in children])
    cod = cat2ty(tree['cat'])
    if tree['type'] == 'ba':
        box = BA(dom[1:])
    elif tree['type'] == 'fa':
        box = FA(dom[:1])
    elif tree['type'] == 'fc':
        box = FC(dom[:1], dom[1:])
    else:
        box = Box(tree['type'], dom, cod)
    return Id(Ty()).tensor(*children) >> box
