# -*- coding: utf-8 -*-

"""
A categorial grammar is a free closed category with words as boxes.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Word

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        cat2ty
        tree2diagram
"""

import re

from discopy import closed, grammar


class Word(grammar.Word, closed.Box):
    """
    A word is a closed box with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.

    Parameters:
        name (str) : The name of the word.
        cod (closed.Ty) : The grammatical type of the word.
        dom (closed.Ty) : An optional domain for the word, empty by default.
    """


def cat2ty(string: str) -> closed.Ty:
    """
    Translate the string representation of a CCG category into DisCoPy.

    Parameters:
        string : The string with slashes representing a CCG category.
    """
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
    return closed.Ty(left)


def tree2diagram(tree: dict, dom=closed.Ty()) -> closed.Diagram:
    """
    Translate a depccg.Tree in JSON format into DisCoPy.

    Parameters:
        tree : The tree to translate.
        dom : The domain for the word boxes, empty by default.
    """
    if 'word' in tree:
        return Word(tree['word'], cat2ty(tree['cat']), dom=dom)
    children = list(map(tree2diagram, tree['children']))
    dom = closed.Ty().tensor(*[child.cod for child in children])
    cod = cat2ty(tree['cat'])
    if tree['type'] == 'ba':
        box = closed.BA(dom.inside[1])
    elif tree['type'] == 'fa':
        box = closed.FA(dom.inside[0])
    elif tree['type'] == 'fc':
        box = closed.FC(dom.inside[0], dom.inside[1])
    else:
        box = closed.Box(tree['type'], dom, cod)
    return closed.Id().tensor(*children) >> box
