# -*- coding: utf-8 -*-

"""
An abstract categorial grammar is a free closed category with words as
constants, i.e. the derivations are linear lambda terms and the lexicons
are functors between free closed categories.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Diagram
    Box
    Eval
    Coeval
    Curry
    Copy
    Discard
    Swap
    Functor
    Lexicon
    TermBase
    Word
    Constant
    Variable
    Application
    Abstraction

Example
-------
>>> from discopy.grammar import categorial
>>> n, s = categorial.Ty("n"), categorial.Ty("s")
>>> Alice, sleeps = n("Alice"), (n >> s)("sleeps")
>>> print(TermBase.from_categorial(Alice(sleeps, left=True)))
n('Alice')((n >> s)('sleeps'), left=True)
"""

from __future__ import annotations

from discopy import closed
from discopy.cat import ob_factory, ar_factory
from discopy.grammar import categorial


@ob_factory
class Ty(closed.Ty):
    """
    An abstract type is a closed type used as the signature of an abstract
    categorial grammar.

    Parameters:
        inside (Ty) : The objects inside the type.
    """
    from_categorial = closed.Ty.from_biclosed


class Exp(closed.Exp):
    "An exponential object in an abstract categorial grammar."

    ob = Ty


@ar_factory
class Diagram(closed.Diagram):
    """
    An abstract diagram is a closed diagram with words as boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty


class Box(closed.Box, Diagram):
    "An abstract box is a closed box in an abstract diagram."


class Eval(closed.Eval, Box):
    "The evaluation of an exponential type."


class Coeval(closed.Coeval, Box):
    "The coevaluation of an exponential type, i.e. the dagger of an Eval."


class Curry(closed.Curry, Box):
    "The currying of an abstract diagram."


class Copy(closed.Copy, Box):
    "The copy of an abstract type, or its discard when ``n=0``."


class Discard(closed.Discard, Copy):
    "The discard of an abstract type, i.e. a copy with zero legs."


class Swap(closed.Swap, Box):
    "The symmetric swap of two abstract types."


class Functor(closed.Functor):
    """
    An abstract functor is a closed functor with abstract diagrams as domain.

    Parameters:
        ob (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


class Lexicon(categorial.Functor, closed.Functor):
    """
    A lexicon is a functor from categorial diagrams to abstract diagrams,
    i.e. it sends words to (almost) linear lambda terms.

    Parameters:
        ob (Mapping[categorial.Ty, Ty]) :
            Map from atomic :class:`categorial.Ty` to :code:`cod.ob`.
        ar (Mapping[categorial.Constant, Term]) :
            Map from words to terms in the codomain.
        cod (Category) : The codomain, :class:`Diagram` by default.

    Example
    -------
    >>> n, s = categorial.Ty("n"), categorial.Ty("s")
    >>> Alice, sleeps = n("Alice"), (n >> s)("sleeps")
    >>> N, S = Ty("N"), Ty("S")
    >>> lexicon = Lexicon(
    ...     ob={n: N, s: S},
    ...     ar={Alice: N("ALICE"), sleeps: (N >> S)("SLEEPS")})
    >>> print(lexicon(Alice(sleeps, left=True)))
    N('ALICE')((N >> S)('SLEEPS'), left=True)
    """
    dom, cod = categorial.Diagram, Diagram


class TermBase(Box, closed.TermBase):
    """
    A term in the internal language of an abstract categorial grammar.
    """
    functor = Functor.id(Diagram)

    @classmethod
    def from_categorial(cls, term: categorial.Term) -> Term:
        """
        Translate a categorial term into an abstract term, dropping planarity
        by collapsing left and right exponentials into a single exponential.

        Parameters:
            term : The categorial term to translate.

        Note
        ----
        Type raising and composition are translated into lambda terms, so the
        result is beta-equivalent to, but not necessarily identical with, the
        image of the simplified term.

        Example
        -------
        >>> from discopy.grammar.categorial import Ty as T
        >>> x, y = T("x"), T("y")
        >>> f, a = (y << x)("f"), x("a")
        >>> print(TermBase.from_categorial(f(a)))
        (x >> y)('f')(x('a'))
        """
        lexicon = Lexicon(
            ob=lambda x: cls.ob(x.name),
            ar=lambda w: Word(w.name, lexicon(w.cod)))
        return lexicon(term)


class Constant(TermBase, closed.Constant):
    "A constant term in an abstract categorial grammar."


class Word(Constant):
    """
    A word is a constant of the abstract signature.

    Parameters:
        name (str) : The name of the word.
        cod (Ty) : The type of the word.
    """


class Variable(TermBase, closed.Variable):
    "A variable term in an abstract categorial grammar."


class Application(TermBase, closed.Application):
    "The application of an abstract term to another."


class Abstraction(TermBase, closed.Abstraction):
    "The abstraction of a variable in an abstract term."


type Term = Constant | Variable | Application | Abstraction

Id = Diagram.id
Diagram.copy_factory = Copy
Diagram.braid_factory = Swap
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.discard_factory = Discard
Ty.exp_factory = Ty.under_factory = Ty.over_factory = staticmethod(Exp)
Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
mapcategorial.TermBase.to_abstract = lambda self: TermBase.from_categorial(self)
