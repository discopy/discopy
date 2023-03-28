# -*- coding: utf-8 -*-

"""
A categorial grammar is a free closed category with words as boxes.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Rule
    Word
    FA
    BA
    FC
    BC
    FX
    BX
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        cat2ty
        tree2diagram
"""

import re

from discopy import closed, messages
from discopy.braided import BinaryBoxConstructor
from discopy.cat import factory, AxiomError
from discopy.grammar import thue
from discopy.utils import assert_isinstance, factory_name, from_tree


@factory
class Diagram(closed.Diagram):
    """
    A categorial diagram is a closed diagram with rules and words as boxes.
    """
    def to_pregroup(self):
        from discopy import rigid
        from discopy.grammar import pregroup

        return Functor(
            ob=lambda x: pregroup.Ty(x.inside[0].name),
            ar=lambda f: pregroup.Box(f.name,
                                      Diagram.to_pregroup(f.dom),
                                      Diagram.to_pregroup(f.cod)),
            cod=rigid.Category(rigid.Ty, pregroup.Diagram))(self)

    @staticmethod
    def fa(left, right):
        """ Forward application. """
        return FA(left << right)

    @staticmethod
    def ba(left, right):
        """ Backward application. """
        return BA(left >> right)

    @staticmethod
    def fc(left, middle, right):
        """ Forward composition. """
        return FC(left << middle, middle << right)

    @staticmethod
    def bc(left, middle, right):
        """ Backward composition. """
        return BC(left >> middle, middle >> right)

    @staticmethod
    def fx(left, middle, right):
        """ Forward crossed composition. """
        return FX(left << middle, right >> middle)

    @staticmethod
    def bx(left, middle, right):
        """ Backward crossed composition. """
        return BX(middle << left, middle >> right)


class Rule(thue.Rule, Diagram):
    """
    A categorial rule is a grammar rule in a categorial diagram.
    """


class Word(thue.Word, Rule):
    """
    A categorial word is a rule with a ``name``, a grammatical type as ``cod``
    and an optional domain ``dom``.

    Parameters:
        name (str) : The name of the word.
        cod (closed.Ty) : The grammatical type of the word.
        dom (closed.Ty) : An optional domain for the word, empty by default.
    """


class Eval(closed.Eval, Rule):
    """
    Evaluation box in a categorial grammar, equivalent to :class:``FA``.
    """


class Curry(closed.Curry, Rule):
    """
    The currying of a categorial diagram.
    """


def unaryBoxConstructor(attr):
    class Constructor:
        @classmethod
        def from_tree(cls, tree):
            return cls(from_tree(tree[attr]))

        def to_tree(self):
            return {
                'factory': factory_name(type(self)),
                attr: getattr(self, attr).to_tree()}
    return Constructor


class FA(unaryBoxConstructor("over"), Rule):
    """ Forward application rule. """
    def __init__(self, over):
        assert_isinstance(over, closed.Over)
        self.over = over
        dom, cod = over @ over.exponent, over.base
        Rule.__init__(self, dom, cod, name=f"FA{over}")

    def __repr__(self):
        return f"FA({repr(self.dom[:1])})"


class BA(unaryBoxConstructor("under"), Rule):
    """ Backward application rule. """
    def __init__(self, under):
        assert_isinstance(under, closed.Under)
        self.under = under
        dom, cod = under.exponent @ under, under.base
        Rule.__init__(self, dom, cod, name=f"BA{under}")

    def __repr__(self):
        return f"BA({repr(self.dom[1:])})"


class FC(BinaryBoxConstructor, Rule):
    """ Forward composition rule. """
    def __init__(self, left, right):
        assert_isinstance(left, closed.Over)
        assert_isinstance(right, closed.Over)
        if left.exponent != right.base:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.exponent, right.base))
        name = f"FC({left}, {right})"
        dom, cod = left @ right, left.base << right.exponent
        Rule.__init__(self, dom, cod, name=name)
        BinaryBoxConstructor.__init__(self, left, right)


class BC(BinaryBoxConstructor, Rule):
    """ Backward composition rule. """
    def __init__(self, left, right):
        assert_isinstance(left, closed.Under)
        assert_isinstance(right, closed.Under)
        if left.base != right.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.base, right.exponent))
        name = f"BC({left}, {right})"
        dom, cod = left @ right, left.exponent >> right.base
        Rule.__init__(self, dom, cod, name=name)
        BinaryBoxConstructor.__init__(self, left, right)


class FX(BinaryBoxConstructor, Rule):
    """ Forward crossed composition rule. """
    def __init__(self, left, right):
        assert_isinstance(left, closed.Over)
        assert_isinstance(right, closed.Under)
        if left.exponent != right.base:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.exponent, right.base))
        name = f"FX({left}, {right})"
        dom, cod = left @ right, right.exponent >> left.base
        Rule.__init__(self, dom, cod, name=name)
        BinaryBoxConstructor.__init__(self, left, right)


class BX(BinaryBoxConstructor, Rule):
    """ Backward crossed composition rule. """
    def __init__(self, left, right):
        assert_isinstance(left, closed.Over)
        assert_isinstance(right, closed.Under)
        if left.base != right.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.base, right.exponent))
        name = f"BX({left}, {right})"
        dom, cod = left @ right, right.base << left.exponent
        Rule.__init__(self, dom, cod, name=name)
        BinaryBoxConstructor.__init__(self, left, right)


class Functor(closed.Functor):
    """
    A categorial functor is a closed functor with a predefined mapping
    for categorial rules.

    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = closed.Category(closed.Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, FA):
            left, right = other.over.left, other.over.right
            return self.cod.ar.fa(self(left), self(right))
        if isinstance(other, BA):
            left, right = other.under.left, other.under.right
            return self.cod.ar.ba(self(left), self(right))
        for cls, method in [(FC, 'fc'), (BC, 'bc')]:
            if isinstance(other, cls):
                left = other.dom.inside[0].left
                middle = other.dom.inside[0].right
                right = other.dom.inside[1].right
                return getattr(self.cod.ar, method)(
                    self(left), self(middle), self(right))
        if isinstance(other, FX):
            left = other.dom.inside[0].left
            middle = other.dom.inside[0].right
            right = other.dom.inside[1].left
            return self.cod.ar.fx(self(left), self(middle), self(right))
        if isinstance(other, BX):
            left = other.dom.inside[0].right
            middle = other.dom.inside[0].left
            right = other.dom.inside[1].right
            return self.cod.ar.bx(self(left), self(middle), self(right))
        return super().__call__(other)


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


def tree2diagram(tree: dict, dom=closed.Ty()) -> Diagram:
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
        box = BA(dom.inside[1])
    elif tree['type'] == 'fa':
        box = FA(dom.inside[0])
    elif tree['type'] == 'fc':
        box = FC(dom.inside[0], dom.inside[1])
    else:
        box = Rule(dom, cod, name=tree['type'])
    return Id().tensor(*children) >> box


Id = Diagram.id
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
