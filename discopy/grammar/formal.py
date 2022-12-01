# -*- coding: utf-8 -*-

"""
A formal grammar is a free monoidal category with words and rules as boxes.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Word
    Rule

Example
-------
>>> s, n, v, p = map(Ty, "SNVP")
>>> r0, r1 = Rule(n @ p, s), Rule(v @ n, p)
>>> Jane, loves, John = Word('Jane', n), Word('loves', v), Word('John', n)
>>> sentence = Jane @ loves @ John >> n @ r1 >> r0
>>> sentence.draw(figsize=(4, 3), path='docs/imgs/grammar/cfg-example.png')

.. image:: /imgs/grammar/cfg-example.png
    :align: center
"""

from discopy import monoidal
from discopy.utils import factory_name


class Word(monoidal.Box):
    """
    A word is a monoidal box with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.

    Parameters:
        name : The name of the word.
        cod : The grammatical type of the word.
        dom : An optional domain for the word, empty by default.
    """
    def __init__(self, name: str, cod: monoidal.Ty, dom: monoidal.Ty = None,
                 **params):
        dom = dom or self.ty_factory()
        super().__init__(name, dom, cod, **params)

    def __repr__(self):
        dom = f", dom={self.dom}" if self.dom else ""
        return factory_name(type(self)) + f"({self.name}, {self.cod}{dom})"


class Rule(monoidal.Box):
    """
    A rule is a box with a monoidal types as ``dom`` and ``cod`` and an
    optional ``name``.

    Parameters:
        dom : The domain of the rule, i.e. its input.
        cod : The codomain of the rule, i.e. its output.
        name : The name of the rule, empty by default.
    """
    def __init__(self, dom: monoidal.Ty, cod: monoidal.Ty, name: str = None,
                 **params):
        super().__init__(name or "", dom, cod, **params)

    def __repr__(self):
        name = f", name={self.name}" if self.name else ""
        return factory_name(type(self)) + f"({self.dom}, {self.cod}{name})"

    __str__ = __repr__
