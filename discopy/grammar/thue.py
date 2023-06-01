"""
A formal grammar is a free monoidal category with words and rules as boxes.

Formal grammars are also known as string rewriting or semi-Thue systems, they
were introduced by Thue :cite:`Thue14`.

The parsing problem is to decide, given two strings, whether there exists a
diagram from one to the other. It has been shown to be undecidable by Post
:cite:`Post47` and Markov :cite:`Markov47`.

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
>>> sentence.draw(figsize=(4, 3), path='docs/_static/grammar/cfg-example.png')

.. image:: /_static/grammar/cfg-example.png
    :align: center
"""
from discopy import monoidal
from discopy.monoidal import Ty  # noqa: F401
from discopy.utils import factory_name


class Rule(monoidal.Box):
    """
    A rule is a box with monoidal types as ``dom`` and ``cod`` and an
    optional ``name``.

    Parameters:
        dom : The domain of the rule, i.e. its input.
        cod : The codomain of the rule, i.e. its output.
        name : The name of the rule, empty by default.
    """
    def __init__(self, dom: monoidal.Ty, cod: monoidal.Ty, name: str = None,
                 **params):
        name = f"Rule({dom}, {cod})" if name is None else name
        monoidal.Box.__init__(self, name, dom, cod, **params)

    def __repr__(self):
        name = f", name={self.name!r}" if self.name else ""
        return factory_name(type(self)) + f"({self.dom!r}, {self.cod!r}{name})"


class Word(Rule):
    """
    A word is a rule with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.

    Parameters:
        name : The name of the word.
        cod : The grammatical type of the word.
        dom : An optional domain for the word, empty by default.
    """
    def __init__(self, name: str, cod: monoidal.Ty, dom: monoidal.Ty = None,
                 **params):
        dom = self.ty_factory() if dom is None else dom
        Rule.__init__(self, dom=dom, cod=cod, name=name, **params)

    def __repr__(self):
        dom = f", dom={repr(self.dom)}" if self.dom else ""
        return factory_name(type(self))\
            + f"({repr(self.name)}, {repr(self.cod)}{dom})"
