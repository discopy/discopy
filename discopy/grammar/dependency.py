# -*- coding: utf-8 -*-

"""
Implements the free coloured operad (multicategory) and its algebras.

See ../docs/notebooks/operads.ipynb for further documentation.
"""

from unittest.mock import Mock

from discopy import monoidal
from discopy.monoidal import Ty
from discopy.grammar.cfg import Word, Rule, Tree


def from_spacy(doc, word_types=False):
    """ Interface with SpaCy's dependency parser """
    root = find_root(doc)
    return doc2tree(root)


def find_root(doc):
    for token in doc:
        if token.dep_ == 'ROOT':
            return token


def doc2tree(root):
    if not root.children:
        return Word(root.text, Ty(root.dep_))
    dom = Ty().tensor(*[Ty(child.dep_) for child in root.children])
    box = Rule(dom, Ty(root.dep_), name=root.text)
    return box(*[doc2tree(child) for child in root.children])
