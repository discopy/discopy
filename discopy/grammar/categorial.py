# -*- coding: utf-8 -*-

"""
A categorial grammar is a free biclosed category with words as boxes.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    TermBase
    Constant
    Variable
    Application
    Abstraction
    FA
    BA
    FC
    BC
    FX
    BX
    Diagram
    Box
    Word
    ForwardCrossedComposition
    BackwardCrossedComposition
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        cat2ty
        tree2diagram
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from discopy import biclosed, messages
from discopy.cat import ar_factory
from discopy.grammar import thue
from discopy.biclosed import Ty, Over, Under
from discopy.utils import (
    assert_isinstance,
    BinaryBoxConstructor,
    AxiomError
)


@ar_factory
class Diagram(biclosed.Diagram):
    """
    A categorial diagram is a biclosed diagram with rules and words as boxes.
    """
    def to_pregroup(self):
        from discopy.grammar import pregroup

        return Functor(
            ob=lambda x: pregroup.Ty(x.inside[0].name),
            ar=lambda f: pregroup.Box(f.name,
                                      Diagram.to_pregroup(f.dom),
                                      Diagram.to_pregroup(f.cod)),
            cod=pregroup.Diagram)(self)

    @staticmethod
    def fa(left, right):
        """ Forward application. """
        return Diagram.eval_factory(left << right, left=True)

    @staticmethod
    def ba(left, right):
        """ Backward application. """
        return Diagram.eval_factory(left >> right, left=False)

    @staticmethod
    def fc(left, middle, right):
        """ Forward composition. """
        return (
            Diagram.id(left << middle) @ Diagram.id(middle << right)
            @ Diagram.id(right)
            >> Diagram.id(left << middle)
            @ Diagram.eval_factory(middle << right, left=True)
            >> Diagram.eval_factory(left << middle, left=True)
        ).curry(left=True)

    @staticmethod
    def bc(left, middle, right):
        """ Backward composition. """
        return (
            Diagram.id(left) @ Diagram.id(left >> middle)
            @ Diagram.id(middle >> right)
            >> Diagram.eval_factory(left >> middle, left=False)
            @ Diagram.id(middle >> right)
            >> Diagram.eval_factory(middle >> right, left=False)
        ).curry()

    @staticmethod
    def fx(left, middle, right):
        """ Forward crossed composition. """
        return ForwardCrossedComposition(left << middle, right >> middle)

    @staticmethod
    def bx(left, middle, right):
        """ Backward crossed composition. """
        return BackwardCrossedComposition(middle << left, middle >> right)


class Box(biclosed.Box, Diagram):
    """
    A categorial box is a grammar rule in a categorial diagram.
    """


class Word(thue.Word, Box):
    """
    A categorial word is a rule with a ``name``, a grammatical type as ``cod``
    and an optional domain ``dom``.

    Parameters:
        name (str) : The name of the word.
        cod (biclosed.Ty) : The grammatical type of the word.
        dom (biclosed.Ty) : An optional domain for the word, empty by default.
    """


class Eval(biclosed.Eval, Box):
    """
    Evaluation box in a categorial grammar.
    """


class Curry(biclosed.Curry, Box):
    """
    The currying of a categorial diagram.
    """


class ForwardCrossedComposition(BinaryBoxConstructor, Box):
    """ Forward crossed composition rule. """
    def __init__(self, left, right):
        assert_isinstance(left, Over)
        assert_isinstance(right, Under)
        if left.exponent != right.base:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.exponent, right.base))
        name = f"ForwardCrossedComposition({left}, {right})"
        dom, cod = left @ right, right.exponent >> left.base
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class BackwardCrossedComposition(BinaryBoxConstructor, Box):
    """ Backward crossed composition rule. """
    def __init__(self, left, right):
        assert_isinstance(left, Over)
        assert_isinstance(right, Under)
        if left.base != right.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                left, right, left.base, right.exponent))
        name = f"BackwardCrossedComposition({left}, {right})"
        dom, cod = left @ right, right.base << left.exponent
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class Functor(biclosed.Functor):
    """
    A categorial functor is a biclosed functor with a predefined mapping
    for categorial rules.

    Parameters:
        ob (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, ForwardCrossedComposition):
            left = other.dom.inside[0].left
            middle = other.dom.inside[0].right
            right = other.dom.inside[1].left
            return self.cod.fx(self(left), self(middle), self(right))
        if isinstance(other, BackwardCrossedComposition):
            left = other.dom.inside[0].right
            middle = other.dom.inside[0].left
            right = other.dom.inside[1].right
            return self.cod.bx(self(left), self(middle), self(right))
        return super().__call__(other)


class TermBase(biclosed.TermBase):
    """
    A term in the internal language of a categorial grammar.
    """
    def __call__(self, other, left=True):
        return FA(self, other) if left else BA(other, self)

    def __lshift__(self, other):
        return FA(self, other)

    def __rshift__(self, other):
        return BA(self, other)

    def to_diagram(self, category=Diagram, box_factory=Word):
        return super().to_diagram(category=category, box_factory=box_factory)


@dataclass(frozen=True)
class Constant(biclosed.Constant, TermBase):
    cod: Ty
    name: str

    def to_diagram(self, category=Diagram, box_factory=Word):
        return box_factory(self.name, dom=self.cod.ob(), cod=self.cod)

    def simplify(self):
        return self


@dataclass(frozen=True)
class Variable(biclosed.Variable, TermBase):
    cod: Ty
    name: str

    def simplify(self):
        return self


@dataclass(frozen=True)
class Abstraction(biclosed.Abstraction, TermBase):
    var: Variable
    body: Term
    left: bool = False

    def simplify(self):
        return Abstraction(self.var, self.body.simplify(), self.left)


class FA(TermBase, biclosed.Application):
    def __init__(self, func, args):
        biclosed.Application.__init__(self, func, args)

    def simplify(self):
        return FA(self.func.simplify(), self.args.simplify())


class BA(TermBase, biclosed.Application):
    def __init__(self, args, func):
        biclosed.Application.__init__(self, func, args, left=False)

    def simplify(self):
        return BA(self.args.simplify(), self.func.simplify())


@dataclass(frozen=True)
class TypeRaising(TermBase):
    base: Ty
    child: Term

    @property
    def freevars(self):
        return self.child.freevars

    def __str__(self):
        return f"{type(self).__name__}({self.base}, {self.child})"

    def to_diagram(self, **kwargs):
        return self.simplify().to_diagram(**kwargs)


class FTR(TypeRaising):
    @property
    def cod(self):
        return self.base << (self.child.cod >> self.base)

    def simplify(self):
        return (self.child.cod >> self.base)(
            lambda f: self.child.simplify() >> f)


class BTR(TypeRaising):
    @property
    def cod(self):
        return (self.base << self.child.cod) >> self.base

    def simplify(self):
        return (self.base << self.child.cod)(
            lambda f, left=True: f << self.child.simplify())


@dataclass(frozen=True)
class BinaryTerm(TermBase):
    left: Term
    right: Term

    def to_diagram(self, **kwargs):
        return self.simplify().to_diagram(**kwargs)

    def __post_init__(self):
        if set(self.left.freevars).intersection(self.right.freevars):
            raise ValueError("Expected disjoint free variables.")

    @property
    def freevars(self):
        return self.left.freevars + self.right.freevars

    def __str__(self):
        return f"{type(self).__name__}({self.left}, {self.right})"

    def simplify(self):
        return type(self)(self.left.simplify(), self.right.simplify())

    def to_diagram(self):
        return self.simplify().to_diagram()


@dataclass(frozen=True)
class FC(BinaryTerm):
    """ Forward composition term, i.e. ``(A << B)(B << C) -> (A << C)``. """
    def __post_init__(self):
        super().__post_init__()
        assert_isinstance(self.left.cod, Over)
        assert_isinstance(self.right.cod, Over)
        if self.right.cod.base != self.left.cod.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self.left.cod, self.right.cod,
                self.left.cod.exponent, self.right.cod.base))

    @property
    def cod(self):
        return self.left.cod.base << self.right.cod.exponent

    def simplify(self):
        f, g = self.left.simplify(), self.right.simplify()
        return self.right.cod.exponent(lambda x: f << (g << x))


@dataclass(frozen=True)
class BC(BinaryTerm):
    """ Backward composition term, i.e. ``(A >> B)(B >> C) -> (A >> C)``. """
    def __post_init__(self):
        super().__post_init__()
        assert_isinstance(self.left.cod, Under)
        assert_isinstance(self.right.cod, Under)
        if self.left.cod.base != self.right.cod.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self.left.cod, self.right.cod,
                self.left.cod.base, self.right.cod.exponent))

    @property
    def cod(self):
        return self.left.cod.exponent >> self.right.cod.base

    def simplify(self):
        f, g = self.left.simplify(), self.right.simplify()
        return self.left.cod.exponent(lambda x, left=True: (x >> f) >> g)


@dataclass(frozen=True)
class FX(BinaryTerm):
    """ Forward crossed composition term, i.e. ``A/B C\\B -> C\\A``. """
    def __post_init__(self):
        super().__post_init__()
        assert_isinstance(self.left.cod, Over)
        assert_isinstance(self.right.cod, Under)
        if self.left.cod.exponent != self.right.cod.base:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self.left.cod, self.right.cod,
                self.left.cod.exponent, self.right.cod.base))

    @property
    def cod(self):
        return self.right.cod.exponent >> self.left.cod.base

    def to_diagram(self, category=Diagram, **kwargs):
        left, middle = self.left.cod.base, self.left.cod.exponent
        right = self.right.cod.exponent
        return self.left.to_diagram(category=category, **kwargs)\
            @ self.right.to_diagram(category=category, **kwargs)\
            >> category.fx(left, middle, right)


@dataclass(frozen=True)
class BX(BinaryTerm):
    """ Backward crossed composition term, i.e. ``B/A B\\C -> A/C``. """
    def __post_init__(self):
        super().__post_init__()
        assert_isinstance(self.left.cod, Over)
        assert_isinstance(self.right.cod, Under)
        if self.left.cod.base != self.right.cod.exponent:
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self.left.cod, self.right.cod,
                self.left.cod.base, self.right.cod.exponent))

    @property
    def cod(self):
        return self.right.cod.base << self.left.cod.exponent

    def to_diagram(self, category=Diagram, **kwargs):
        left, middle = self.left.cod.exponent, self.left.cod.base
        right = self.right.cod.base
        return self.left.to_diagram(category=category, **kwargs)\
            @ self.right.to_diagram(category=category, **kwargs)\
            >> category.bx(left, middle, right)


type Term = (
    Constant | Variable | Abstraction
    | FA | BA | FC | BC | FX | BX)


def cat2ty(string: str) -> Ty:
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
    return Ty(left)


def tree2diagram(tree: dict, dom=Ty()) -> Diagram:
    """
    Translate a depccg.Tree in JSON format into DisCoPy.

    Parameters:
        tree : The tree to translate.
        dom : The domain for the word boxes, empty by default.
    """
    if 'word' in tree:
        return Word(tree['word'], cat2ty(tree['cat']), dom=dom)
    children = list(map(tree2diagram, tree['children']))
    dom = Ty().tensor(*[child.cod for child in children])
    cod = cat2ty(tree['cat'])
    if tree['type'] == 'ba':
        rule = Diagram.ba(dom[:1], dom.inside[1].base)
    elif tree['type'] == 'fa':
        rule = Diagram.fa(dom.inside[0].base, dom[1:])
    elif tree['type'] == 'fc':
        rule = Diagram.fc(
            dom.inside[0].base,
            dom.inside[0].exponent,
            dom.inside[1].exponent)
    else:
        rule = Box(tree['type'], dom, cod)
    return Id().tensor(*children) >> rule


Id = Diagram.id
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
