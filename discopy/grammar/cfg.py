# -*- coding: utf-8 -*-

"""
A context free rule is a box with one output.
A context free grammar is a diagram with rules as boxes.


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

from __future__ import annotations

import nltk

from discopy import cat, monoidal, grammar
from discopy.cat import factory, Category, Functor, AxiomError
from discopy.monoidal import Ty, assert_isatomic
from discopy.utils import assert_isinstance, factory_name
from discopy.grammar import thue


@factory
class Tree:
    """
    A tree is a rule for the ``root`` and a list of trees called ``branches``.

    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Rule(x @ x, x, name='f'), Rule(x @ y, x, name='g')
    >>> h = Rule(y @ x, x, name='h')
    >>> assert f(g, h) == Tree(f, *[g, h])

    The axioms of multicategories hold on the nose.

    >>> assert Id(x)(f) == f == f(Id(x), Id(x))
    >>> left = f(Id(x), h)(g, Id(x), Id(x))
    >>> right = f(g, Id(x))(Id(x), Id(x), h)
    >>> assert f(g, h) == left == right
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
        return "{}({})".format(self.root.name,
                               ', '.join(map(Tree.__str__, self.branches)))

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


class Rule(Tree, thue.Rule):
    """
    A rule is a generator of free operads, given by an atomic monoidal type
    for ``dom``, a monoidal type for ``cod`` and an optional ``name``.
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


class Word(thue.Word, Rule):
    """
    A word is a monoidal box with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.

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
    def __init__(self, dom):
        self.dom, self.cod = dom, dom
        Rule.__init__(self, dom, dom, name="Id({})".format(dom))

    def __repr__(self):
        return "Id({})".format(self.dom)


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
        ob : The mapping from domain to codomain colours.
        ar : The mapping from domain to codomain operations.
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


rule2box = lambda node: monoidal.Box(node.name, node.dom, node.cod)


def tree2diagram(tree, contravariant=False):
    """
    Interface between Tree and monoidal.Diagram.

    >>> x = Ty('x')
    >>> f = Rule(x @ x, x, name='f')
    >>> tree = f(f(f, f), f)
    >>> print(tree2diagram(tree))
    f @ x @ x @ x @ x >> x @ f @ x @ x >> f @ x @ x >> x @ f >> f
    """
    if isinstance(tree, Rule):
        return rule2box(tree)
    return rule2box(tree.root) << monoidal.Id().tensor(
        *[tree2diagram(branch) for branch in tree.branches])


def from_nltk(tree, lexicalised=True, word_types=False):
    """
    Interface with NLTK

    >>> t = nltk.Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
    >>> print(from_nltk(t))
    S(I, VP(saw, him))
    >>> from_nltk(t).branches[0]
    grammar.cfg.Word('I', monoidal.Ty(cat.Ob('NP')))
    """
    branches = []
    for branch in tree:
        if isinstance(branch, str):
            return Word(branch, Ty(tree.label()))
        else:
            branches += [from_nltk(branch)]
    label = tree.label()
    dom = Ty().tensor(*[Ty(branch.label()) for branch in tree])
    root = Rule(dom, Ty(label), name=label)
    return root(*branches)
