# -*- coding: utf-8 -*-

"""
A context free grammar is a formal grammar where the rules all have a codomain
of length 1.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Tree
    Rule
    Word
    Id
    Operad
    Algebra

Axioms
------

The axioms of multicategories (aka operads) hold on the nose.

>>> x, y = Ty('x'), Ty('y')
>>> f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
>>> h = Rule(y @ x, x, name='h')
>>> assert f(g, h) == Tree(f, *[g, h])

>>> assert Id(x)(f) == f == f(Id(x), Id(x))
>>> left = f(Id(x), h)(g, Id(x), Id(x))
>>> right = f(g, Id(x))(Id(x), Id(x), h)
>>> assert f(g, h) == left == right
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from discopy import monoidal
from discopy.cat import factory, Category, Functor
from discopy.grammar import thue
from discopy.monoidal import Ty
from discopy.utils import (
    assert_isinstance, factory_name, assert_isatomic, AxiomError)

if TYPE_CHECKING:
    import nltk


@factory
class Tree:
    """
    A tree is a rule for the ``root`` and a list of trees called ``branches``.

    Example
    -------

    We build a syntax tree from a context-free grammar.

    >>> n, d, v = Ty('N'), Ty('D'), Ty('V')
    >>> vp, np, s = Ty('VP'), Ty('NP'), Ty('S')
    >>> Caesar, crossed = Word('Caesar', n), Word('crossed', v)
    >>> the, Rubicon = Word('the', d), Word('Rubicon', n)
    >>> VP, NP = Rule(n @ v, vp), Rule(d @ n, np)
    >>> S = Rule(vp @ np, s)
    >>> sentence = S(VP(Caesar, crossed), NP(the, Rubicon))
    """
    ty_factory = Ty

    def __init__(self, root: Rule, *branches: Tree):
        assert_isinstance(root, Rule)
        for branch in branches:
            assert_isinstance(branch, Tree)
        if not isinstance(self, Rule) and not root.dom == Ty().tensor(
                *[branch.cod for branch in branches]):
            raise AxiomError
        self.cod, self.root, self.branches = root.cod, root, branches

    def __repr__(self):
        return factory_name(type(self)) + f"({self.root}, *{self.branches})"

    def __str__(self):
        if isinstance(self, Rule):
            return self.name
        return self.root.name\
            + f"({', '.join(map(Tree.__str__, self.branches))})"

    def __call__(self, *others):
        if not others or all([isinstance(other, Id) for other in others]):
            return self
        if isinstance(self, Id):
            return others[0]
        if isinstance(self, Rule):
            return Tree(self, *others)
        if isinstance(self, Tree):
            lengths = [len(branch.dom) for branch in self.branches]
            ranges = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]
            branches = [self.branches[i](*others[ranges[i]:ranges[i + 1]])
                        for i in range(len(self.branches))]
            return Tree(self.root, *branches)
        raise NotImplementedError()

    @staticmethod
    def id(dom):
        return Id(dom)

    def __eq__(self, other):
        return self.root == other.root and self.branches == other.branches

    def to_diagram(self, contravariant=False) -> monoidal.Diagram:
        """
        Interface between Tree and monoidal.Diagram.

        >>> x = Ty('x')
        >>> f = Rule(x @ x, x, name='f')
        >>> tree = f(f(f, f), f)
        >>> print(tree.to_diagram().foliation())
        f @ f @ x @ x >> f @ f >> f
        """
        return self.root.to_diagram()\
            << monoidal.Id().tensor(*[t.to_diagram() for t in self.branches])

    @staticmethod
    def from_nltk(tree: nltk.Tree, lexicalised=True, word_types=False) -> Tree:
        """
        Interface with NLTK

        >>> import nltk
        >>> t = nltk.Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
        >>> print(Tree.from_nltk(t))
        S(I, VP(saw, him))
        >>> Tree.from_nltk(t).branches[0]
        grammar.cfg.Word('I', monoidal.Ty(cat.Ob('NP')))
        """
        branches = []
        for branch in tree:
            if isinstance(branch, str):
                return Word(branch, Ty(tree.label()))
            else:
                branches += [Tree.from_nltk(branch)]
        label = tree.label()
        dom = Ty().tensor(*[Ty(branch.label()) for branch in tree])
        root = Rule(dom, Ty(label), name=label)
        return root(*branches)


class Rule(Tree, thue.Rule):
    """
    A rule is a generator of free operads, given by an atomic type ``dom``,
    a type ``cod`` of arbitrary length and an optional ``name``.
    """
    def __init__(self, dom: monoidal.Ty, cod: monoidal.Ty, name: str = None):
        assert_isinstance(dom, Ty)
        assert_isatomic(cod, Ty)
        thue.Rule.__init__(self, dom=dom, cod=cod, name=name)
        Tree.__init__(self, root=self)

    def __eq__(self, other):
        if isinstance(other, Rule):
            return self.dom == other.dom and self.cod == other.cod \
                and self.name == other.name
        if isinstance(other, Tree):
            return other.root == self and other.branches == []

    def to_diagram(self) -> monoidal.Box:
        return monoidal.Box(self.name, self.dom, self.cod)


class Word(thue.Word, Rule):
    """
    A word is a leaf in a context-free tree.

    Parameters:
        name : The name of the word.
        cod : The grammatical type of the word.
        dom : An optional domain for the word, empty by default.
    """
    def __init__(self, name: str, cod: monoidal.Ty, dom: monoidal.Ty = Ty(),
                 **params):
        thue.Word.__init__(self, name=name, dom=dom, cod=cod, **params)
        Rule.__init__(self, dom=dom, cod=cod, name=name, **params)


class Id(Rule):
    """ The identity is a rule that does nothing. """
    def __init__(self, dom):
        self.dom, self.cod = dom, dom
        Rule.__init__(self, dom, dom, name=f"Id({dom})")

    def __repr__(self):
        return f"Id({self.dom})"


class Operad(Category):
    """
    An operad is a category with a method ``__call__`` which constructs a tree
    from a root and a list of branches.

    Parameters:
        ob : The colours of the operad.
        ar : The operations of the operad.
    """
    ob = Ty
    ar = Tree


class Algebra(Functor):
    """
    An algebra is a functor with the free operad as domain and a given operad
    as codomain.

    Parameters:
        ob (dict[monoidal.Ty, cod.ob]) :
            The mapping from domain to codomain colours.
        ar (dict[Rule, cod.ar]):
            The mapping from domain to codomain operations.
        cod (Operad) : The codomain of the algebra.
    """
    dom = cod = Operad()

    def __call__(self, other):
        if isinstance(other, Id):
            return self.cod.id(self.ob[other.dom])
        if isinstance(other, Rule):
            return self.ar[other]
        if isinstance(other, Tree):
            return self(other.root)(*map(self, other.branches))
        raise TypeError
