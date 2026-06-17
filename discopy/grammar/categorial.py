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
        return Diagram.eval_factory(left << right)

    @staticmethod
    def ba(left, right):
        """ Backward application. """
        return Diagram.eval_factory(left >> right)

    @staticmethod
    def fc(left, middle, right):
        """ Forward composition. """
        return (
            Diagram.id(left << middle)
            @ Diagram.eval_factory(middle << right)
            >> Diagram.eval_factory(left << middle)
        ).curry(left=True)

    @staticmethod
    def bc(left, middle, right):
        """ Backward composition. """
        return (
            Diagram.eval_factory(left >> middle)
            @ Diagram.id(middle >> right)
            >> Diagram.eval_factory(middle >> right)
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


class TermBase(Box, biclosed.TermBase):
    """
    A term in the internal language of a categorial grammar.
    """
    functor = Functor.id(Diagram)
    freevars = None

    def __call__(self, other, left=False):
        return BA(self, other) if left else FA(self, other)


class Constant(TermBase, biclosed.Constant):
    def __init__(self, name: str, cod: Ty):
        biclosed.Constant.__init__(self, name, cod)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    def simplify(self):
        return self


class Variable(TermBase, biclosed.Variable):
    def simplify(self):
        return self


class Abstraction(TermBase, biclosed.Abstraction):
    var: Variable
    body: Term
    left: bool = False

    def __init__(self, var: Variable, body: Term, left: bool = False):
        biclosed.Abstraction.__init__(self, var, body, left)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    def simplify(self):
        return Abstraction(self.var, self.body.simplify(), self.left)


class FA(TermBase, biclosed.Application):
    "Application of type ``Y`` with subterms of type ``Y << X`` and ``X``."
    def __init__(self, func, args):
        biclosed.Application.__init__(self, func, args, left=False)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    def simplify(self):
        return self.func.simplify()(self.args.simplify())


class BA(TermBase, biclosed.Application):
    "Application of type ``Y`` with subterms of type ``X`` and ``X >> Y``."
    def __init__(self, args, func):
        biclosed.Application.__init__(self, func, args, left=True)
        TermBase.__init__(self, self.name, self.dom, self.cod)

    def simplify(self):
        return self.args.simplify()(self.func.simplify(), left=True)


class TypeRaising(TermBase):
    "Abstract superclass of :class:`FTR` and :class:`BTR`."
    base: Ty
    child: Term

    def __init__(self, base, child, cod):
        name = f"{type(self).__name__}({base}, {child})"
        self.base, self.child, self.freevars = base, child, child.freevars
        super().__init__(name, child.dom, cod)

    def eval(self, **kwargs):
        return self.simplify().eval(**kwargs)


class FTR(TypeRaising):
    "Forward type raising ``Y << (X >> Y)`` with base ``Y`` and child ``X``."
    def __init__(self, base, child):
        super().__init__(base, child, base << (child.cod >> base))

    def simplify(self):
        f = Variable("f", self.child.cod >> self.base)
        return Abstraction(f, self.child.simplify()(f, left=True))


class BTR(TypeRaising):
    "Backward type raising ``(Y << X) >> Y`` with base ``Y`` and child ``X``."
    def __init__(self, base, child):
        super().__init__(base, child, (base << child.cod) >> base)

    def simplify(self):
        f = Variable("f", self.base << self.child.cod)
        return Abstraction(f, f(self.child.simplify()), left=True)


@dataclass(frozen=True)
class BinaryTerm(TermBase):
    "Abstract superclass of :class:`FC`, :class:`BC`, :class:`FX`, :class:`BX`"
    left: Term
    right: Term

    def __post_init__(self):
        if set(self.left.freevars).intersection(self.right.freevars):
            raise ValueError("Expected disjoint free variables.")
        object.__setattr__(
            self, "freevars", self.left.freevars + self.right.freevars)
        object.__setattr__(self, "dom", self.left.dom + self.right.dom)

    def __str__(self):
        return f"{type(self).__name__}({self.left}, {self.right})"

    def simplify(self):
        return type(self)(self.left.simplify(), self.right.simplify())

    def eval(self, **kwargs):
        return self.simplify().eval(**kwargs)


@dataclass(frozen=True)
class FC(BinaryTerm):
    "Forward composition ``A << C`` with subterms ``A << B`` and ``B << C``. "
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
        x = Variable("x", self.right.cod.exponent)
        return Abstraction(x, f(g(x)))


@dataclass(frozen=True)
class BC(BinaryTerm):
    "Backward composition ``A >> C`` with subterms ``A >> B`` and ``B >> C``."
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
        x = Variable("x", self.left.cod.exponent)
        return Abstraction(x, x(f, left=True)(g, left=True), left=True)


@dataclass(frozen=True)
class FX(BinaryTerm):
    "Forward crossing ``A >> C`` with subterms ``B << A`` and ``B >> C``."
    def __post_init__(self):
        super().__post_init__()
        ForwardCrossedComposition(self.left.cod, self.right.cod)

    @property
    def cod(self):
        return self.right.cod.exponent >> self.left.cod.base

    def eval(self, functor=None):
        functor, X = functor or self.functor, self.left.cod.base
        Y, Z = self.left.cod.exponent, self.right.cod.exponent
        f, g = self.left.eval(functor), self.right.eval(functor)
        return f @ g >> functor.cod.fx(*map(functor, [X, Y, Z]))


@dataclass(frozen=True)
class BX(BinaryTerm):
    "Backward crossing ``A << C`` with subterms ``A << B`` and ``C >> B``."
    def __post_init__(self):
        super().__post_init__()
        BackwardCrossedComposition(self.left.cod, self.right.cod)

    @property
    def cod(self):
        return self.right.cod.base << self.left.cod.exponent

    def eval(self, functor=None):
        functor, Z = functor or self.functor, self.right.cod.base
        X, Y = self.left.cod.exponent, self.left.cod.base
        f, g = self.left.eval(functor), self.right.eval(functor)
        return f @ g >> functor.cod.bx(*map(functor, [X, Y, Z]))


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
