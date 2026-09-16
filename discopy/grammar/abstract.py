# -*- coding: utf-8 -*-

"""
Abstract categorial grammars after de Groote's `Towards abstract categorial
grammars (2001) <https://aclanthology.org/P01-1033/>`_: a vocabulary
generates a free closed category and a :class:`Lexicon` is a functor between
two of them, which is all there is to a grammar.

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

Vocabularies and lexicons
-------------------------

A *vocabulary* is a higher-order signature: atomic types and constants, each
with an implicative type ``x >> y`` built from the atoms, e.g. the words of a
language with their grammatical types. It generates a free closed category,
whose morphisms are the lambda terms of its internal language, here
:class:`Term`. Nothing forces a term to be linear: the closed categories of
DisCoPy are markov by design, so a variable may be copied and discarded, as
the variables of ground type are in
the semantics of the example below; the paper's linear terms are the special
case where every variable occurs once, see ``is_linear``. A :class:`Lexicon`
from one vocabulary to another sends atomic types to types and constants to
terms of the image of their types: it is a :class:`Functor` between two free
closed categories, and lexicons compose. A lexicon is all there is to a
grammar: its abstract vocabulary is the atoms and constants it is defined on,
its abstract language the closed terms over them and its object language their
image. The paper's distinguished type ``s`` picks the sentences out of the
abstract language, the input of a parser to come.

Strings
-------

Strings are the paper's, section 4: one atomic type :data:`Position` and a
string is a map from positions to positions, a term of type
``String = Position >> Position``. A word is a constant of that type, the
empty string is the identity ``Position(lambda x: x)`` and concatenation is
composition, ``John.compose(seeks)`` for *John seeks*, see
:meth:`discopy.closed.TermBase.compose`: the paper writes it
``lambda z. x (y z)`` with the first word applied last, here the first word
is applied first, so a string reads its positions in order.

Example
-------

The two readings of Montague's *John seeks a unicorn* share one syntax, with
a constant for each reading of *seeks* in the abstract vocabulary:

>>> n, np, s = map(Ty, ("n", "np", "s"))
>>> J, U, A = np("J"), n("U"), (n >> np)("A")
>>> S_re, S_dicto = ((np >> (np >> s))(S) for S in ("S_re", "S_dicto"))
>>> John, seeks, a, unicorn = map(
...     String, ("John", "seeks", "a", "unicorn"))
>>> seek = String(lambda x: String(lambda y: x.compose(seeks, y)))
>>> syntax = Lexicon(
...     ob_map={n: String, np: String, s: String},
...     ar_map={J: John, U: unicorn, A: String(lambda x: a.compose(x)),
...             S_re: seek, S_dicto: seek})
>>> from discopy.python import Function
>>> words = Functor(
...     {Position: list}, lambda w: lambda: lambda xs: xs + [w.name],
...     cod=Function)
>>> for reading in (S_re(J)(A(U)), S_dicto(J)(A(U))):
...     print(*words(syntax(reading))()([]))
John seeks a unicorn
John seeks a unicorn

The paper's semantic lexicon reads noun phrases as quantifiers. The image of
*a* copies its variable of ground type, so the semantic terms are not linear,
and the two readings are two closed terms of type ``t``:

>>> e, t = Ty("e"), Ty("t")
>>> Predicate, Quantifier = e >> t, (e >> t) >> t
>>> JOHN, UNICORN = e("JOHN"), Predicate("UNICORN")
>>> TRY_TO, FIND = (e >> Quantifier)("TRY_TO"), (e >> Predicate)("FIND")
>>> exists, and_ = Quantifier("exists"), (t >> (t >> t))("and")
>>> some = Predicate(lambda P: Predicate(lambda Q: exists(
...     e(lambda x: and_(P(x))(Q(x))))))
>>> seek_re = Quantifier(lambda P: Quantifier(lambda Q: Q(e(lambda x: P(
...     e(lambda y: TRY_TO(y)(e(lambda z: FIND(z)(x)))))))))
>>> seek_dicto = Quantifier(lambda P: Quantifier(lambda Q: P(e(lambda x:
...     TRY_TO(x)(e(lambda y: Q(e(lambda z: FIND(y)(z)))))))))
>>> semantics = Lexicon(
...     ob_map={n: Predicate, np: Quantifier, s: t},
...     ar_map={J: Predicate(lambda P: P(JOHN)), U: UNICORN,
...             A: some, S_re: seek_re, S_dicto: seek_dicto})
>>> assert not some.is_linear
>>> de_re, de_dicto = semantics(S_re(J)(A(U))), semantics(S_dicto(J)(A(U)))
>>> assert de_re != de_dicto and de_re.cod == de_dicto.cod == t
>>> assert not de_re.freevars and not de_dicto.freevars

Categorial grammars
-------------------

A categorial grammar is a vocabulary of its own, a planar one: a lexicon out
of it is a :class:`discopy.grammar.categorial.Functor` into abstract terms,
sending each atom to a type and each word to a term of the image of its type,
its strings written per word like every lexicon of the paper.
:meth:`Diagram.from_categorial` forgets planarity instead, collapsing left and
right exponentials into the one exponential of a closed category, so that a
derivation becomes a closed term: application evaluates, composition and type
raising curry, and crossed composition composes.

>>> from discopy.grammar import categorial
>>> N, S = categorial.Ty("n"), categorial.Ty("s")
>>> Alice, loves, Bob = N("Alice"), ((N >> S) << N)("loves"), N("Bob")
>>> LOVES = String("loves")
>>> strings = categorial.Functor(
...     ob_map={N: String, S: String},
...     ar_map={Alice: String("Alice"), Bob: String("Bob"),
...             loves: String(lambda o: String(
...                 lambda x: x.compose(LOVES, o)))},
...     cod=Diagram)
>>> print(*words(strings(Alice(loves(Bob), left=True)))()([]))
Alice loves Bob
>>> print(Alice(loves(Bob), left=True).to_abstract())
(n >> (n >> s))('loves')(n('Bob'))(n('Alice'))
"""

from __future__ import annotations

from discopy import closed, cmap, hypergraph
from discopy.cat import factory
from discopy.grammar import categorial


@factory
class Ty(closed.Ty):
    "An implicative type, the type of an abstract categorial grammar."

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
        collapse, constants stay constants and words become boxes, crossed
        compositions become compositions and the composition and type-raising
        terms become lambda terms.

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
                box.name, functor(box.cod))
            if isinstance(box, categorial.Constant)
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
    terms of the image of their types, section 2.2 of de Groote's `Towards
    abstract categorial grammars (2001) <https://aclanthology.org/P01-1033/>`_.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from atomic types to types.
        ar_map (Mapping[Constant, Term]) : Map from constants to terms.

    Lexicons compose: ``first >> second`` sends a constant ``c`` to
    ``second(first(c))``, with the target vocabulary of the first lexicon
    serving as the source vocabulary of the second.

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

Position = Ty("o")
""" The atomic type of positions in a string, an arbitrary atom. """

String = Position >> Position
"""
The type of strings, maps from positions to positions: a word is a constant
of type ``String``, ``Position(lambda x: x)`` is the empty string and
``John.compose(seeks)`` their concatenation, see
:meth:`discopy.closed.TermBase.compose`.
"""
