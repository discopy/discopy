# -*- coding: utf-8 -*-

"""
An abstract categorial grammar is a free closed category with words as
constants, after de Groote's `Towards abstract categorial grammars (2001)
<https://aclanthology.org/P01-1033/>`_: derivations are (almost) linear
lambda terms and lexicons are functors between free closed categories.

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
    Permutation
    Swap
    Trace
    Sum
    Functor
    Lexicon
    TermBase
    Constant
    Variable
    Application
    Abstraction

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        concat
        slots
        saturate

Vocabularies and lexicons
-------------------------

A *vocabulary* is a higher-order linear signature: atomic types and constants,
each with a linear implicative type ``x >> y`` built from the atoms, e.g. the
words of a language with their grammatical types. It generates a free closed
category, whose morphisms are the lambda terms of its internal language, here
:class:`Term`: linear in the paper, with copying and discarding allowed here
as the category is markov. A :class:`Lexicon` from one vocabulary to another
sends atomic types to types and constants to terms of the image of their
types: it is a :class:`Functor` between two free closed categories, and
lexicons compose. An abstract categorial grammar is a lexicon together with
a distinguished type ``s``: its abstract language is the set of closed terms
of type ``s``, its object language is their image under the lexicon.

Strings
-------

Strings are encoded with one atomic type ``*``: a string is a term of type
``string = * >> *``, the empty string is the identity, concatenation is
composition and a word is a constant of type ``string``, see :func:`concat`.
A categorial grammar comes with a string lexicon that reads the word order
off its slashes, see :meth:`Lexicon.from_categorial`, while the abstract
term of one of its derivations drops planarity, see
:meth:`Diagram.from_categorial`.

Example
-------

The two readings of Montague's *John seeks a unicorn* share one syntax, with
a constant for each reading of *seeks* in the abstract vocabulary:

>>> n, np, s = map(Ty, ("n", "np", "s"))
>>> J, U, A = np("J"), n("U"), (n >> np)("A")
>>> S_re, S_dicto = ((np >> (np >> s))(S) for S in ("S_re", "S_dicto"))
>>> John, seeks, a, unicorn = map(
...     string, ("John", "seeks", "a", "unicorn"))
>>> seek = string(lambda x: string(lambda y: concat(x, seeks, y)))
>>> syntax = Lexicon(
...     ob_map={n: string, np: string, s: string},
...     ar_map={J: John, U: unicorn, A: string(lambda x: concat(a, x)),
...             S_re: seek, S_dicto: seek})
>>> for reading in (S_re(J)(A(U)), S_dicto(J)(A(U))):
...     print(*[word.name for word in syntax(reading).normal_form().constants])
John seeks a unicorn
John seeks a unicorn
"""

from __future__ import annotations

from discopy import closed, cmap, hypergraph
from discopy.cat import factory
from discopy.grammar import categorial
from discopy.utils import AxiomError


@factory
class Ty(closed.Ty):
    "A linear implicative type, the type of an abstract categorial grammar."

    @classmethod
    def from_categorial(cls, old: categorial.Ty) -> Ty:
        """
        Translate a categorial type into an abstract type, collapsing left
        and right exponentials into a single exponential.

        Parameters:
            old : The categorial type to translate.
        """
        return cls.from_biclosed(old)


class Exp(closed.Exp):
    "An exponential object in an abstract categorial grammar."

    ob = Ty


@factory
class Diagram(closed.Diagram):
    """
    An abstract diagram is a closed diagram with words as boxes, i.e. a
    derivation in an abstract categorial grammar.

    The rules of a categorial grammar are its closed structure: application
    is evaluation, composition is the currying of two evaluations and crossed
    composition coincides with composition, since a closed category has one
    exponential.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty

    @classmethod
    def fa(cls, left: Ty, right: Ty) -> Diagram:
        "Forward application ``(left << right) @ right -> left``."
        return cls.ev(left, right, left=True)

    @classmethod
    def ba(cls, left: Ty, right: Ty) -> Diagram:
        "Backward application ``left @ (left >> right) -> right``."
        return cls.ev(right, left, left=False)

    @classmethod
    def fc(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        """
        Forward composition
        ``(left << middle) @ (middle << right) -> left << right``.
        """
        return (cls.id(left << middle) @ cls.fa(middle, right)
                >> cls.fa(left, middle)).curry(len(right), left=True)

    @classmethod
    def bc(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        """
        Backward composition
        ``(left >> middle) @ (middle >> right) -> left >> right``.
        """
        return (cls.ba(left, middle) @ cls.id(middle >> right)
                >> cls.ba(middle, right)).curry(len(left), left=False)

    @classmethod
    def fx(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        "Forward crossed composition, which coincides with :meth:`fc`."
        return cls.fc(left, middle, right)

    @classmethod
    def bx(cls, left: Ty, middle: Ty, right: Ty) -> Diagram:
        "Backward crossed composition, which coincides with :meth:`bc`."
        return cls.bc(left, middle, right)

    @classmethod
    def from_categorial(cls, diagram: categorial.Diagram) -> Diagram:
        """
        The abstract diagram of a categorial diagram, or the abstract term of
        a categorial term, dropping planarity: left and right exponentials
        collapse, words become constants, crossed compositions become
        compositions and the composition and type-raising terms become
        lambda terms.

        Parameters:
            diagram : The categorial diagram or term to translate.

        Example
        -------
        >>> from discopy.grammar import categorial
        >>> n, s = categorial.Ty("n"), categorial.Ty("s")
        >>> Alice, loves, Bob = n("Alice"), ((n >> s) << n)("loves"), n("Bob")
        >>> print(Diagram.from_categorial(Alice(loves(Bob), left=True)))
        (n >> (n >> s))('loves')(n('Bob'))(n('Alice'))
        """
        functor = categorial.Functor(
            ob_map=lambda x: cls.ob(x.inside[0].name),
            ar_map=lambda box: cls.ob.constant_factory(
                box.name, functor(box.cod)) if not box.dom
            else Box(box.name, functor(box.dom), functor(box.cod)),
            cod=cls)
        return functor(diagram)


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


class Permutation(closed.Permutation, Box):
    "A permutation of abstract types."


class Swap(Permutation, closed.Swap, Box):
    "The symmetric swap of two abstract types."


class Trace(closed.Trace, Box):
    "A trace in an abstract categorial grammar."


class Sum(closed.Sum, Box):
    "A formal sum of abstract diagrams."


class Functor(closed.Functor):
    """
    An abstract functor is a closed functor with abstract diagrams as domain.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


class Lexicon(Functor):
    """
    A lexicon is a functor from one vocabulary to another, i.e. between two
    free closed categories: it sends atomic types to types and constants to
    terms of the image of their types, section 2.2 of the paper.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic types to types.
        ar_map (Mapping[Constant, Term]) : Map from constants to terms.

    Lexicons compose: ``first >> second`` sends a constant ``c`` to
    ``second(first(c))``, so that the object language of one grammar is the
    abstract language of the next.

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> first = Lexicon(ob_map={x: y}, ar_map={x("a"): y("b")})
    >>> second = Lexicon(ob_map={y: z}, ar_map={y("b"): z("c")})
    >>> assert first(x("a")) == y("b") and (first >> second)(x("a")) == z("c")
    >>> Lexicon(ob_map={x: y}, ar_map={x("a"): (y >> y)("b")})(x("a"))
    Traceback (most recent call last):
        ...
    discopy.utils.AxiomError: Expected a term of type y for x('a'), got ...
    """
    dom = cod = Diagram

    def __call__(self, other):
        result = super().__call__(other)
        if isinstance(other, Constant) and result.cod != self(other.cod):
            raise AxiomError(
                f"Expected a term of type {self(other.cod)} for {other}, "
                f"got {result} of type {result.cod}.")
        return result

    @classmethod
    def from_categorial(cls, *words: categorial.Word) -> Lexicon:
        """
        The string lexicon of a categorial grammar, section 4 of the paper:
        every atomic type is a :data:`string`, every word is concatenated
        with its arguments in the order its slashes give, see :func:`slots`.

        Parameters:
            words : The words of the categorial grammar, i.e. constants.

        Note
        ----
        The word order of a derivation is read off its slashes, so a crossed
        composition, whose surface order breaks them by design, comes out in
        the order that its types give.

        Example
        -------
        >>> from discopy.grammar import categorial
        >>> n, s = categorial.Ty("n"), categorial.Ty("s")
        >>> Alice, loves, Bob = n("Alice"), ((n >> s) << n)("loves"), n("Bob")
        >>> strings = Lexicon.from_categorial(Alice, loves, Bob)
        >>> sentence = strings(Diagram.from_categorial(
        ...     Alice(loves(Bob), left=True)))
        >>> print(*[word.name for word in sentence.normal_form().constants])
        Alice loves Bob
        """
        return cls(ob_map=lambda _: string, ar_map={
            Diagram.from_categorial(word): slots(string(word.name), word.cod)
            for word in words})


CMap = cmap.CMap[Diagram]


class TermBase(Box, closed.TermBase):
    "A term in the internal language of an abstract categorial grammar."
    functor = Functor.id(Diagram)


class Constant(TermBase, closed.Constant):
    "A constant of the vocabulary of an abstract categorial grammar."


Word = Constant


class Variable(TermBase, closed.Variable):
    "A variable term in an abstract categorial grammar."


class Application(TermBase, closed.Application):
    "The application of an abstract term to another."


class Abstraction(TermBase, closed.Abstraction):
    "The abstraction of a variable in an abstract term."


type Term = Constant | Variable | Application | Abstraction

Id = Diagram.id
Diagram.functor_factory = Functor
Diagram.copy_factory = Copy
Diagram.permutation_factory = Permutation
Diagram.swap_factory = Swap
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.trace_factory = Trace
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Ty.exp_factory = Ty.under_factory = Ty.over_factory = staticmethod(Exp)
Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
Hypergraph = hypergraph.Hypergraph[Diagram]

star = Ty("*")
""" The atomic type of strings, an arbitrary atom with strings as maps. """

string = star >> star
""" The type of strings, i.e. of words and of their concatenations. """


def concat(*terms: Term) -> Term:
    """
    The concatenation of string terms, i.e. their composition
    ``*(lambda x: s(t(x)))``, the empty string ``*(lambda x: x)`` when there
    are none.

    Parameters:
        terms : Terms of type :data:`string`, e.g. the words of a sentence.

    Example
    -------
    >>> Alice, loves, Bob = map(string, ("Alice", "loves", "Bob"))
    >>> print(concat(Alice, loves, Bob))
    *(lambda x: (* >> *)('Alice')((* >> *)('loves')((* >> *)('Bob')(x))))
    >>> assert concat(concat(Alice, loves), Bob).normal_form()\\
    ...     == concat(Alice, loves, Bob)\\
    ...     == concat(Alice, concat(loves, Bob)).normal_form()
    """
    var = Variable.fresh("x", star, *terms)
    body = var
    for term in reversed(terms):
        body = term(body)
    return Abstraction(var, body)


def slots(word: Term, ty: categorial.Ty) -> Term:
    """
    The string term of a word of a categorial type: the word concatenated
    with a variable for each of its arguments, on the right of ``y << x``
    and on the left of ``x >> y``, a higher-order argument being
    :func:`saturate` d with empty strings first.

    Parameters:
        word : A term of type :data:`string`.
        ty : The categorial type of the word.

    Example
    -------
    >>> from discopy.grammar import categorial
    >>> n, s = categorial.Ty("n"), categorial.Ty("s")
    >>> loves = slots(string("loves"), (n >> s) << n)
    >>> Alice, Bob = string("Alice"), string("Bob")
    >>> sentence = loves(Bob)(Alice).normal_form()
    >>> print(*[word.name for word in sentence.constants])
    Alice loves Bob
    """
    types = categorial.Functor(ob_map=lambda _: string, ar_map={}, cod=Diagram)
    if not ty.is_exp:
        return word
    var = Variable.fresh("v", types(ty.exponent), word)
    argument = saturate(var, ty.exponent)
    result = concat(word, argument) if ty.is_over else concat(argument, word)
    return Abstraction(var, slots(result, ty.base))


def saturate(term: Term, ty: categorial.Ty) -> Term:
    """
    The string of a term of the string type of a categorial type, i.e. the
    term applied to the empty string for each of its arguments.

    Parameters:
        term : A term of the string type of ``ty``.
        ty : A categorial type.

    Example
    -------
    >>> from discopy.grammar import categorial
    >>> n, s = categorial.Ty("n"), categorial.Ty("s")
    >>> sleeps = slots(string("sleeps"), n >> s)
    >>> assert saturate(sleeps, n >> s).normal_form()\\
    ...     == concat(string("sleeps"))
    """
    if ty.is_exp:
        return saturate(term(slots(concat(), ty.exponent)), ty.base)
    return term
