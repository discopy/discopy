# -*- coding: utf-8 -*-

"""
Implements the free coloured operad (multicategory) and its algebras.

See ../docs/notebooks/operads.ipynb for further documentation.
"""

from discopy import monoidal
from discopy.monoidal import Ty
from discopy.grammar.cfg import Rule, Tree


def from_spacy(doc, word_types=False):
    """ Interface with SpaCy dependency parser """
    root = find_root(doc)
    return doc2tree(root, word_types=word_types)


def find_root(doc):
    for word in doc:
        if word.dep_ == 'ROOT':
            return word


def doc2tree(root, word_types=False):
    children = list(root.children)
    if not children:
        return Rule(root.text, Ty(root.dep_), [Ty(root.text)]) \
            if word_types else Rule(root.text, Ty(root.dep_), [])
    box = Rule(root.text, Ty(root.dep_),
              [Ty(child.dep_) for child in children])
    return box(*[doc2tree(child, word_types=word_types)
                 for child in children])
