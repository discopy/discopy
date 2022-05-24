# -*- coding: utf-8 -*-

"""
Implements the free coloured operad (multicategory) and its algebras.

See ../docs/notebooks/operads.ipynb for further documentation.
"""

from discopy.cat import Ob, Quiver
from discopy import cat, monoidal
import nltk


class Tree:
    """
    We can check the axioms of multicategories hold.

    >>> x, y = Ob('x'), Ob('y')
    >>> f, g, h = Box('f', x, [x, x]), Box('g', x, [x, y]), Box('h', x, [y, x])
    >>> assert Id(x)(f) == f == f(Id(x), Id(x))
    >>> left = f(Id(x), h)(g, Id(x), Id(x))
    >>> middle = f(g, h)
    >>> right = f(g, Id(x))(Id(x), Id(x), h)
    >>> assert left == middle == right == Tree(root=f, branches=[g, h])
    """
    def __init__(self, root, branches, _scan=True):
        if not isinstance(root, Box):
            raise TypeError()
        if not all([isinstance(branch, Tree) for branch in branches]):
            raise TypeError()
        if _scan and not root.cod == [branch.dom for branch in branches]:
            raise AxiomError()
        self.dom, self.root, self.branches = root.dom, root, branches

    @property
    def cod(self):
        if isinstance(self, Box):
            return self._cod
        else:
            return [x for x in branch.cod for branch in self.branches]

    def __repr__(self):
        return "Tree({}, {})".format(self.root, self.branches)

    def __str__(self):
        if isinstance(self, Box):
            return self.name
        return "{}({})".format(self.root.name,
                               ', '.join(map(Tree.__str__, self.branches)))

    def __call__(self, *others):
        if not others or all([isinstance(other, Id) for other in others]):
            return self
        if isinstance(self, Id):
            return others[0]
        if isinstance(self, Box):
            return Tree(self, list(others))
        if isinstance(self, Tree):
            lengths = [len(branch.cod) for branch in self.branches]
            ranges = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]
            branches = [self.branches[i](*others[ranges[i]:ranges[i + 1]])
                        for i in range(len(self.branches))]
            return Tree(self.root, branches, _scan=False)
        raise NotImplementedError()

    @staticmethod
    def id(dom):
        return Id(dom)

    def __eq__(self, other):
        return self.root == other.root and self.branches == other.branches


class Box(Tree):
    """
    Implements generators of free operads, i.e. the nodes in the trees.

    We build a syntax tree from a context-free grammar.

    >>> n, d, v = Ob('N'), Ob('D'), Ob('V')
    >>> vp, np, s = Ob('VP'), Ob('NP'), Ob('S')
    >>> Caesar, crossed = Box('Caesar', n, []), Box('crossed', v, []),
    >>> the, Rubicon = Box('the', d, []), Box('Rubicon', n, [])
    >>> VP, NP = Box('VP', vp, [n, v]), Box('NP', np, [d, n])
    >>> S = Box('S', s, [vp, np])
    >>> sentence = S(VP(Caesar, crossed), NP(the, Rubicon))
    """
    def __init__(self, name, dom, cod):
        if not (isinstance(dom, Ob) and isinstance(cod, list)
                and all([isinstance(x, Ob) for x in cod])):
            return TypeError
        self.name, self.dom, self._cod = name, dom, cod
        Tree.__init__(self, self, [], _scan=False)

    def __repr__(self):
        return "Box('{}', {}, {})".format(self.name, self.dom, self._cod)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, Box):
            return self.dom == other.dom and self.cod == other.cod \
                and self.name == other.name
        if isinstance(other, Tree):
            return other.root == self and other.branches == []


class Id(Box):
    def __init__(self, dom):
        self.dom, self._cod = dom, [dom]
        Box.__init__(self, "Id({})".format(dom), dom, dom)

    def __repr__(self):
        return "Id({})".format(self.dom)


class Algebra:
    """ Algebras of free operads. """
    def __init__(self, ob, ar, cod=Tree, contravariant=False):
        self.cod, self.contravariant = cod, contravariant
        self.ob = Quiver(ob) if callable(ob) else ob
        self.ar = Quiver(ar) if callable(ar) else ar

    def __call__(self, tree):
        if isinstance(tree, Id):
            return self.cod.id(self.ob[tree.dom])
        if isinstance(tree, Box):
            return self.ar[tree]
        box = self.ar[tree.root]
        if isinstance(box, monoidal.Diagram):
            if self.contravariant:
                return box << monoidal.Diagram.tensor(
                    *[self(branch) for branch in tree.branches])
            return box >> monoidal.Diagram.tensor(
                *[self(branch) for branch in tree.branches])
        return box(*[self(branch) for branch in tree.branches])


ob2ty = lambda ob: monoidal.Ty(ob)
node2box = lambda node: monoidal.Box(node.name, monoidal.Ty(node.dom),
                                     monoidal.Ty(*node.cod))
t2d = Algebra(ob2ty, node2box, cod=monoidal.Diagram)
node2boxc = lambda node: monoidal.Box(node.name, monoidal.Ty(*node.cod),
                                      monoidal.Ty(node.dom))
t2dc = Algebra(ob2ty, node2boxc, cod=monoidal.Diagram, contravariant=True)


def tree2diagram(tree, contravariant=False):
    """
    Interface between operad.Tree and monoidal.Diagram.
    
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    >>> doc = nlp("Fifty four was my number")
    >>> tree2diagram(from_spacy(doc)).cod
    Ty()
    >>> tree2diagram(from_spacy(doc), contravariant=True).cod
    Ty('ROOT')
    """
    if contravariant:
        return t2dc(tree)
    return t2d(tree)


def from_nltk(tree, lexicalised=True, word_types=False):
    """
    Interface with NLTK

    >>> t = nltk.Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
    >>> print(from_nltk(t))
    S(I, VP(saw, him))
    >>> print(from_nltk(t, lexicalised=False))
    S(NP(I), VP(V(saw), NP(him)))
    >>> from_nltk(t, word_types=True).branches[0]
    Box('I', NP, [Ob('I')])
    """
    if lexicalised:
        branches, cod = [], []
        for branch in tree:
            if isinstance(branch, str):
                return Box(branch, Ob(tree.label()), [Ob(branch)])\
                    if word_types else Box(branch, Ob(tree.label()), [])
            else:
                branches += [from_nltk(branch, word_types=word_types)]
                cod += [Ob(branch.label())]
        root = Box(tree.label(), Ob(tree.label()), cod)
        return root(*branches)
    else:
        if isinstance(tree, str):
            return Box(tree, Ob(tree), [Ob(tree)]) \
                if word_types else Box(tree, Ob(tree), [])
        else:
            cod = [Ob(branch) if isinstance(branch, str)
                   else Ob(branch.label()) for branch in tree]
            return Box(tree.label(), Ob(tree.label()), cod)(
                *[from_nltk(branch, lexicalised=False, word_types=word_types)
                  for branch in tree])


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
        return Box(root.text, Ob(root.dep_), [Ob(root.text)]) \
            if word_types else Box(root.text, Ob(root.dep_), [])
    box = Box(root.text, Ob(root.dep_),
              [Ob(child.dep_) for child in children])
    return box(*[doc2tree(child, word_types=word_types)
                 for child in children])
