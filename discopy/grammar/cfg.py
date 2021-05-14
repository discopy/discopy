# -*- coding: utf-8 -*-

"""
Implements context-free grammars.

>>> s, n, v, vp = Ty('S'), Ty('N'), Ty('V'), Ty('VP')
>>> R0, R1 = Box('R0', vp @ n, s), Box('R1', n @ v , vp)
>>> Jane, loves = Word('Jane', n), Word('loves', v)
>>> cfg = CFG(R0, R1, Jane, loves)
>>> gen = cfg.generate(start=s, max_sentences=2, max_depth=6)
>>> for sentence in gen: print(sentence)
Jane >> loves @ Id(N) >> Jane @ Id(V @ N) >> R1 @ Id(N) >> R0
Jane >> loves @ Id(N) >> Jane @ Id(V @ N) >> R1 @ Id(N) >> R0
>>> gen = cfg.generate(
...     start=s, max_sentences=2, max_depth=6,
...     remove_duplicates=True, max_iter=10)
>>> for sentence in gen: print(sentence)
Jane >> loves @ Id(N) >> Jane @ Id(V @ N) >> R1 @ Id(N) >> R0

>>> sentence.draw(figsize=(4, 3),\\
... path='docs/_static/imgs/grammar/cfg-example.png')

.. image:: ../_static/imgs/grammar/cfg-example.png
    :align: center
"""

import random

from discopy import messages
from discopy.monoidal import Ty, Box, Id


class Word(Box):
    """
    Implements words as boxes with a :class:`discopy.monoidal.Ty` as codomain.

    >>> from discopy.rigid import Ty
    >>> Alice = Word('Alice', Ty('n'))
    >>> loves = Word('loves',
    ...     Ty('n').r @ Ty('s') @ Ty('n').l)
    >>> Alice
    Word('Alice', Ty('n'))
    >>> loves
    Word('loves', Ty(Ob('n', z=1), 's', Ob('n', z=-1)))
    """
    def __init__(self, name, cod, dom=None, data=None, _dagger=False, _z=0):
        if not isinstance(name, str):
            raise TypeError(messages.type_err(str, name))
        if not isinstance(cod, Ty):
            raise TypeError(messages.type_err(Ty, cod))
        dom = dom or cod[0:0]
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        super().__init__(
            name, dom, cod, data=data, _dagger=_dagger, _z=_z)

    def __repr__(self):
        return "Word({}, {}{})".format(
            repr(self.name), repr(self.cod),
            ", dom={}".format(repr(self.dom)) if self.dom else "")


class CFG:
    """
    Context-free grammar.
    """
    def __init__(self, *productions):
        self._productions = productions

    @property
    def productions(self):
        """
        Production rules, i.e. boxes with :class:`discopy.monoidal.Ty`
        as dom and cod.
        """
        return self._productions

    def __repr__(self):
        return "CFG{}".format(repr(self._productions))

    def generate(self, start, max_sentences, max_depth, max_iter=100,
                 remove_duplicates=False, not_twice=None, seed=None):
        """
        Generate sentences from a context-free grammar.
        Assumes the only terminal symbol is :code:`Ty()`.

        Parameters
        ----------

        start : type
            root of the generated trees.
        max_sentences : int
            maximum number of sentences to generate.
        max_depth : int
            maximum depth of the trees.
        max_iter : int
            maximum number of iterations, set to 100 by default.
        remove_duplicates : bool
            if set to True only distinct syntax trees will be generated.
        not_twice : list
            list of productions that you don't want appearing twice
            in a sentence, set to the empty list by default
        """
        if seed is not None:
            random.seed(seed)
        prods, cache = list(self.productions), set()
        n_sentences, i = 1, 0
        while n_sentences <= (max_sentences or n_sentences) and i < max_iter:
            i += 1
            sentence = Id(start)
            depth = 0
            while depth < max_depth:
                recall = depth
                if sentence.dom == Ty():
                    if remove_duplicates and sentence in cache:
                        break
                    yield sentence
                    if remove_duplicates:
                        cache.add(sentence)
                    n_sentences += 1
                    break
                tag = sentence.dom[0]
                random.shuffle(prods)
                for prod in prods:
                    if prod in (not_twice or []) and prod in sentence.boxes:
                        continue
                    if Ty(tag) == prod.cod:
                        sentence = sentence << prod @ Id(sentence.dom[1:])
                        depth += 1
                        break
                if recall == depth:  # in this case, no production was found
                    break
