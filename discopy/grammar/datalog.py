# -*- coding: utf-8 -*-

"""
Parsing and generation of abstract categorial grammars as Datalog queries,
following Kanazawa's *Parsing and generation as Datalog queries* (ACL 2007)
and *Parsing and generation as Datalog query evaluation* (IfCoLog 2017).

The lexicon of a second-order abstract categorial grammar is translated into
a Datalog program with one rule per word, the input term into a database and
a query, so that the derivations of the input are in bijection with the
derivation trees for the query. The translation is a closed functor into
:class:`Query`, hypergraphs where spiders are Datalog variables (or database
constants) and boxes are atoms: the image of an almost linear lambda term is
its principal typing, with Kanazawa's type flattening given by
``exp(base, exponent) == base @ exponent``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Query
    Functor
    Atom
    Rule
    Semiring
    Program

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        identify
        parse

Example
-------
>>> from discopy.grammar import categorial
>>> from discopy.grammar.abstract import Ty, Lexicon
>>> n, s = categorial.Ty("n"), categorial.Ty("s")
>>> Alice, sleeps = n("Alice"), (n >> s)("sleeps")
>>> e, t = Ty("e"), Ty("t")
>>> ALICE, SLEEPS = e("ALICE"), (e >> t)("SLEEPS")
>>> lexicon = Lexicon(
...     ob_map={n: e, s: t}, ar_map={Alice: ALICE, sleeps: SLEEPS})
>>> print(next(parse(lexicon, SLEEPS(ALICE), s)))
n('Alice')((n >> s)('sleeps'), left=True)
"""

from __future__ import annotations

import math

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from discopy import cat, monoidal, biclosed, closed, frobenius
from discopy.grammar import categorial
from discopy.utils import AxiomError, MappingOrCallable


class Query(frobenius.Hypergraph):
    """
    A Datalog query encoded as a hypergraph: spiders are variables (or
    database constants), boxes are atoms and the codomain is the output.

    A query is also a morphism in a closed category where the exponential
    is given by the tensor, i.e. Kanazawa's type flattening: this way the
    principal typing of an almost linear lambda term is its image under a
    closed :class:`Functor` with queries as codomain.

    Example
    -------
    >>> x, y = frobenius.Ty('x'), frobenius.Ty('y')
    >>> assert Query.exp(x, y) == x @ y
    >>> assert Query.ev(x, y).dom == x @ y @ y and Query.ev(x, y).cod == x
    """
    exp = over = under = staticmethod(lambda base, exponent: base @ exponent)

    @classmethod
    def ev(cls, base: frobenius.Ty, exponent: frobenius.Ty,
           left=True) -> Query:
        """
        The evaluation of an exponential, i.e. the pairwise fusion of the
        exponent wires of a function with the wires of its argument.

        Parameters:
            base : The flattened base of the exponential.
            exponent : The flattened exponent of the exponential.
            left : Whether the argument comes from the right.
        """
        result = cls.id(base) @ cls.spiders(2, 0, exponent)
        return result if left\
            else cls.swap(exponent, base @ exponent) >> result

    def curry(self, n=1, left=True) -> Query:
        """
        Move the last ``n`` wires of the domain to the end of the codomain.

        Parameters:
            n : The number of wires to curry.
            left : Whether to curry on the left, the only supported case.

        Note
        ----
        Unlike :meth:`biclosed.Diagram.curry`, the parameter ``n`` counts
        wires, i.e. atomic sorts, rather than types.

        Example
        -------
        >>> x, y = frobenius.Ty('x'), frobenius.Ty('y')
        >>> f = Query.from_box(frobenius.Box('f', x @ y, x))
        >>> assert f.curry().dom == x and f.curry().cod == x @ y
        >>> assert f.curry() @ Query.id(y) >> Query.ev(x, y) == f
        """
        if not left:
            raise NotImplementedError
        dom = self.dom[:len(self.dom) - n]
        exponent = self.dom[len(self.dom) - n:]
        return type(self).id(dom) @ type(self).spiders(0, 2, exponent)\
            >> self @ type(self).id(exponent)

    @classmethod
    def discard(cls, typ: frobenius.Ty) -> Query:
        "Discard a flattened type, i.e. spiders with no output."
        return cls.spiders(1, 0, typ)

    @property
    def atoms(self) -> tuple[Atom, ...]:
        "One :class:`Atom` for each box, with spiders as arguments."
        return tuple(
            Atom(box.name, tuple(dom_wires) + tuple(cod_wires))
            for box, (dom_wires, cod_wires)
            in zip(self.boxes, self.wires[1]))


@dataclass(frozen=True)
class Atom:
    """
    A predicate applied to a tuple of arguments, i.e. spider identifiers
    read as Datalog variables in a rule or as constants in a database.

    Parameters:
        pred : The name of the predicate.
        args : The arguments of the atom.

    Example
    -------
    >>> assert Atom('f', (0, 1)).substitute({0: 2, 1: 3}) == Atom('f', (2, 3))
    """
    pred: str
    args: tuple[int, ...]

    def substitute(self, env: dict[int, int]) -> Atom:
        "Apply a substitution to the arguments of the atom."
        return Atom(self.pred, tuple(env[arg] for arg in self.args))

    def unify(self, fact: Atom, env: dict[int, int]) -> Optional[dict]:
        """
        Extend a substitution so that it sends the atom to a ground fact,
        or return ``None`` if there is a clash.

        Parameters:
            fact : The ground fact to match against.
            env : The substitution to extend.
        """
        if self.pred != fact.pred or len(self.args) != len(fact.args):
            return None
        env = dict(env)
        for var, val in zip(self.args, fact.args):
            if env.setdefault(var, val) != val:
                return None
        return env


class Functor(closed.Functor):
    """
    A closed functor from almost linear lambda terms to Datalog queries:
    atomic types are sent to one-wire sorts, exponentials to flattened
    tensors, constants to boxes and shared atomic variables to spiders,
    so that the image of a term is its principal typing.

    Parameters:
        ob : Map from atomic types to :class:`frobenius.Ty` sorts,
             the name-preserving map by default.

    Example
    -------
    >>> from discopy.grammar.abstract import Ty
    >>> e, t = Ty("e"), Ty("t")
    >>> F = Functor()
    >>> assert F(e >> t) == frobenius.Ty('t', 'e')
    >>> assert F((e >> t) >> t) == frobenius.Ty('t', 't', 'e')
    """
    dom, cod = closed.Diagram, Query

    def __init__(self, ob=None):
        super().__init__(
            ob or (lambda x: frobenius.Ty(x.name)), ar_map={}, cod=Query)

    def __call__(self, other):
        if isinstance(other, biclosed.TermBase):
            return self.encode(other, ())
        if isinstance(other, biclosed.Exp):
            return self(other.base) @ self(other.exponent)
        if isinstance(other, monoidal.Ty):
            return frobenius.Ty().tensor(*[self(x) for x in other.inside])
        if isinstance(other, cat.Ob):
            return self.ob_map[other]
        return super().__call__(other)

    def encode(self, term: closed.Term,
               context: tuple[closed.Variable, ...]) -> Query:
        """
        The image of a term in a context, i.e. a query with the flattened
        context as domain and the flattened type of the term as codomain.

        Parameters:
            term : The term to encode.
            context : The variables in scope, in the order of the domain.
        """
        if isinstance(term, biclosed.Variable):
            if term not in context:
                raise ValueError(f"Unbound variable {term}.")
            return Query.id(frobenius.Ty()).tensor(*[
                Query.id(self(var.cod)) if var == term
                else Query.discard(self(var.cod)) for var in context])
        if isinstance(term, biclosed.Constant):
            context_ty = frobenius.Ty().tensor(
                *[self(var.cod) for var in context])
            return Query.discard(context_ty) @ Query.from_box(
                frobenius.Box(term.name, frobenius.Ty(), self(term.cod)))
        if isinstance(term, biclosed.Application):
            for var in set(term.func.freevars) & set(term.args.freevars):
                if len(var.cod) != 1 or var.cod.is_exp:
                    raise AxiomError(
                        "Almost linear terms may only duplicate "
                        f"variables of atomic type, got {var.cod}.")
            context_ty = frobenius.Ty().tensor(
                *[self(var.cod) for var in context])
            return Query.spiders(1, 2, context_ty)\
                >> self.encode(term.func, context)\
                @ self.encode(term.args, context)\
                >> Query.ev(
                    self(term.func.cod.base), self(term.func.cod.exponent))
        if isinstance(term, biclosed.Abstraction):
            return self.encode(term.body, context + (term.var, ))\
                .curry(len(self(term.var.cod)))
        raise TypeError(f"Expected a term, got {type(term)}.")


@dataclass(frozen=True, eq=False)
class Rule:
    """
    The Datalog rule for a word of a second-order abstract categorial
    grammar, i.e. the principal typing of its image under the lexicon.

    Parameters:
        word : The lexical item, kept for decoding derivations.
        head : The atom for the target type of the word.
        children : The atoms for the argument types, outermost first.
        constants : The atoms for the constants in the image of the word.
        slots : The pairs of argument type and direction, outermost first.
        hypergraph : The principal typing as a :class:`Query`.
    """
    word: categorial.Constant
    head: Atom
    children: tuple[Atom, ...]
    constants: tuple[Atom, ...]
    slots: tuple[tuple[categorial.Ty, bool], ...]
    hypergraph: Query

    @property
    def body(self) -> tuple[Atom, ...]:
        "The body of the rule: child atoms followed by constant atoms."
        return self.children + self.constants

    @classmethod
    def from_word(cls, word: categorial.Constant, image: closed.Term,
                  lexicon: cat.Functor, functor: Functor) -> Rule:
        """
        Translate a word and its image under a lexicon into a rule.

        Parameters:
            word : A constant with a second-order categorial type.
            image : The image of the word under the lexicon.
            lexicon : The lexicon, used to translate the argument types.
            functor : The :class:`Functor` used to encode the image.
        """
        ty, slots = word.cod, []
        while ty.is_exp:
            argument = ty.exponent
            if len(argument) != 1 or argument.is_exp:
                raise AxiomError(
                    f"Expected a second-order type, got {word.cod}.")
            slots.append((argument, ty.is_under))
            ty = ty.base
        if len(ty) != 1 or ty.is_exp:
            raise AxiomError(f"Expected an atomic target, got {ty}.")
        hypergraph = functor(image)
        cod_wires = tuple(hypergraph.wires[2])
        target_width = len(functor(lexicon(ty)))
        head = Atom(ty.inside[0].name, cod_wires[:target_width])
        children, position = [], len(cod_wires)
        for argument, _ in slots:
            width = len(functor(lexicon(argument)))
            children.append(Atom(
                argument.inside[0].name,
                cod_wires[position - width:position]))
            position -= width
        return cls(word, head, tuple(children), hypergraph.atoms,
                   tuple(slots), hypergraph)

    def __call__(self, *terms: categorial.Term) -> categorial.Term:
        """
        The decoding of a derivation step: apply the word to the subterms
        derived for each argument, with the direction given by the slots.

        Parameters:
            terms : The subterms derived for each argument.
        """
        result = self.word
        for (_, from_left), term in zip(self.slots, terms):
            result = term(result, left=True) if from_left else result(term)
        return result

    def fire(self, index_total: dict, index_delta: dict, index_older: dict
             ) -> Iterator[tuple[Atom, tuple[Atom, ...]]]:
        """
        Enumerate the ground instances of the rule that use at least one
        fact from the last round, following the semi-naive discipline:
        for each pivot position, the facts at earlier positions come from
        the total, the pivot from the delta and later positions from the
        older facts, so that every instance is enumerated exactly once.

        Parameters:
            index_total : All derived facts, indexed by predicate.
            index_delta : The facts from the last round.
            index_older : The facts from earlier rounds.
        """
        def join(subgoals, sources, env):
            if not subgoals:
                yield env
                return
            for fact in sources[0].get(subgoals[0].pred, ()):
                extended = subgoals[0].unify(fact, env)
                if extended is not None:
                    yield from join(subgoals[1:], sources[1:], extended)

        body = self.body
        for pivot in range(len(body)):
            sources = [index_total] * pivot + [index_delta]\
                + [index_older] * (len(body) - pivot - 1)
            for env in join(body, sources, {}):
                yield self.head.substitute(env), tuple(
                    child.substitute(env) for child in self.children)


def identify(term: closed.Term) -> closed.Term:
    """
    The preprocessing step applied to the input term before building the
    database, the identity by default.

    Kanazawa (2007) identifies distinct occurrences of the same constant
    just in case they occur in the same position within alpha-equivalent
    subterms of atomic type; this is required for input terms where the
    same constant occurs more than once. Pass a custom callable to
    :func:`parse` to plug in such a rule.

    Parameters:
        term : The input term to preprocess.

    Example
    -------
    >>> from discopy.grammar.abstract import Ty
    >>> a = Ty("x")("a")
    >>> assert identify(a) == a
    """
    return term


@dataclass(frozen=True)
class Semiring:
    """
    A commutative semiring in which to evaluate a packed forest, i.e. the
    provenance polynomial of a Datalog query.

    Parameters:
        zero : The unit of addition, the score of an underivable fact.
        one : The unit of multiplication, the score of a database fact.
        plus : The addition, combining alternative derivations.
        times : The multiplication, combining the premises of a rule.

    Note
    ----
    The module provides ``BOOL`` (derivability), ``COUNT`` (number of
    derivations), ``PROB`` (inside probabilities), ``MAXPLUS`` (Viterbi
    scores) and ``LOGPROB`` (log-space inside probabilities). Weights may
    be any objects supporting the semiring operations, e.g. fractions for
    exact arithmetic or autodifferentiation scalars for training.

    Example
    -------
    >>> assert COUNT.plus(COUNT.one, COUNT.one) == 2
    >>> assert MAXPLUS.times(MAXPLUS.one, 42) == 42
    """
    zero: Any
    one: Any
    plus: Callable[[Any, Any], Any]
    times: Callable[[Any, Any], Any]


def _logaddexp(left, right):
    if left == float('-inf'):
        return right
    if right == float('-inf'):
        return left
    top = max(left, right)
    return top + math.log(math.exp(left - top) + math.exp(right - top))


BOOL = Semiring(False, True, lambda x, y: x or y, lambda x, y: x and y)
COUNT = Semiring(0, 1, lambda x, y: x + y, lambda x, y: x * y)
PROB = Semiring(0.0, 1.0, lambda x, y: x + y, lambda x, y: x * y)
MAXPLUS = Semiring(float('-inf'), 0.0, max, lambda x, y: x + y)
LOGPROB = Semiring(float('-inf'), 0.0, _logaddexp, lambda x, y: x + y)


@dataclass
class Program:
    """
    The Datalog program of a second-order abstract categorial grammar,
    with one :class:`Rule` for each word of a finite lexicon.

    Parameters:
        rules : The rules of the program.
        functor : The :class:`Functor` used to encode terms.
        lexicon : The lexicon from which the program was built.
    """
    rules: tuple[Rule, ...]
    functor: Functor
    lexicon: cat.Functor = None

    @classmethod
    def from_lexicon(cls, lexicon: cat.Functor, ob=None) -> Program:
        """
        Translate a finite lexicon into a Datalog program.

        Parameters:
            lexicon : A lexicon with a finite mapping from words to terms.
            ob : An optional map from atomic object types to sorts.
        """
        if not isinstance(lexicon.ar_map.mapping, Mapping):
            raise ValueError("Expected a finite lexicon.")
        functor = Functor(ob)
        rules = tuple(
            Rule.from_word(word, lexicon(word), lexicon, functor)
            for word in lexicon.ar_map.mapping)
        nonterminals = {rule.head.pred for rule in rules}.union(*[
            {child.pred for child in rule.children} for rule in rules])
        for rule in rules:
            for atom in rule.constants:
                if atom.pred in nonterminals:
                    raise ValueError(
                        f"Constant {atom.pred} clashes with a nonterminal.")
        return cls(rules, functor, lexicon)

    def database(self, term: closed.Term, identify: callable = None
                 ) -> tuple[set[Atom], tuple[int, ...]]:
        """
        Translate an input term into a database of ground facts and the
        arguments of the query, i.e. the codomain wires of its encoding.

        Parameters:
            term : The closed term to translate.
            identify : An optional preprocessing step, see :func:`identify`.
        """
        hypergraph = self.functor(
            term if identify is None else identify(term))
        return set(hypergraph.atoms), tuple(hypergraph.wires[2])

    def seminaive(self, facts) -> dict[Atom, list]:
        """
        The proof-relevant semi-naive evaluation of the program on a
        database, following procedure SEMINAIVE-PARSE of Kanazawa (2017):
        the result maps each derivable fact to the list of ground rule
        instances that derive it, i.e. a packed forest encoded as a
        propositional Horn clause program.

        Parameters:
            facts : The ground facts of the database.
        """
        def index(atoms):
            result = {}
            for atom in atoms:
                result.setdefault(atom.pred, []).append(atom)
            return result

        total, delta = set(facts), set(facts)
        forest = {fact: [] for fact in facts}
        while delta:
            index_total, index_delta, index_older = map(
                index, (total, delta, total - delta))
            new = set()
            for rule in self.rules:
                for head, children in rule.fire(
                        index_total, index_delta, index_older):
                    justifications = forest.setdefault(head, [])
                    if (rule, children) not in justifications:
                        justifications.append((rule, children))
                    if head not in total:
                        new.add(head)
            total |= new
            delta = new
        return forest

    def derivations(self, forest: dict[Atom, list], goal: Atom,
                    _path=frozenset()) -> Iterator[categorial.Term]:
        """
        Enumerate the derivations of a goal from a packed forest, decoding
        each derivation tree into a categorial term.

        Parameters:
            forest : The output of :meth:`seminaive`.
            goal : The atom to derive.

        Note
        ----
        Only acyclic derivations are enumerated, i.e. derivations where no
        goal is derived from itself.
        """
        def expand(rule, children, subterms):
            if len(subterms) == len(children):
                yield rule(*subterms)
                return
            child = children[len(subterms)]
            for term in self.derivations(forest, child, _path | {goal}):
                yield from expand(rule, children, subterms + (term, ))

        if goal in _path:
            return
        for rule, children in forest.get(goal, []):
            yield from expand(rule, children, ())

    def inside(self, forest: dict[Atom, list], weights, semiring=PROB,
               passes: int = None) -> dict[Atom, Any]:
        """
        The inside scores of the facts of a packed forest in a semiring,
        i.e. the evaluation of the provenance polynomial: the score of a
        fact is the sum over its justifications of the weight of the rule
        times the product of the scores of the children.

        Parameters:
            forest : The output of :meth:`seminaive`.
            weights : Map from :class:`Rule` to weights in the semiring.
            semiring : The :class:`Semiring` in which to evaluate.
            passes : The number of iterations, the size of the forest by
                     default, which is exact for acyclic forests and a
                     truncation at bounded derivation height otherwise.
        """
        weights = MappingOrCallable(weights)
        passes = len(forest) if passes is None else passes
        scores = {
            fact: semiring.one if not justifications else semiring.zero
            for fact, justifications in forest.items()}
        for _ in range(passes):
            update = {}
            for fact, justifications in forest.items():
                if not justifications:
                    update[fact] = semiring.one
                    continue
                total = semiring.zero
                for rule, children in justifications:
                    value = weights[rule]
                    for child in children:
                        value = semiring.times(
                            value, scores.get(child, semiring.zero))
                    total = semiring.plus(total, value)
                update[fact] = total
            scores = update
        return scores

    def outside(self, forest: dict[Atom, list], goal: Atom, weights,
                semiring=PROB, passes: int = None,
                inside_scores: dict[Atom, Any] = None) -> dict[Atom, Any]:
        """
        The outside scores of the facts of a packed forest with respect
        to a goal, i.e. the derivative of the inside score of the goal
        with respect to the inside score of each fact.

        Parameters:
            forest : The output of :meth:`seminaive`.
            goal : The atom with respect to which to differentiate.
            weights : Map from :class:`Rule` to weights in the semiring.
            semiring : The :class:`Semiring` in which to evaluate.
            passes : The number of iterations, see :meth:`inside`.
            inside_scores : The output of :meth:`inside`, recomputed by
                            default.
        """
        weights = MappingOrCallable(weights)
        passes = len(forest) if passes is None else passes
        if inside_scores is None:
            inside_scores = self.inside(forest, weights, semiring, passes)
        scores = {fact: semiring.zero for fact in forest}
        scores[goal] = semiring.one
        for _ in range(passes):
            update = {fact: semiring.zero for fact in forest}
            update[goal] = semiring.one
            for fact, justifications in forest.items():
                for rule, children in justifications:
                    base = semiring.times(
                        scores.get(fact, semiring.zero), weights[rule])
                    for i, child in enumerate(children):
                        value = base
                        for j, other in enumerate(children):
                            if j != i:
                                value = semiring.times(
                                    value, inside_scores.get(
                                        other, semiring.zero))
                        update[child] = semiring.plus(
                            update.get(child, semiring.zero), value)
            scores = update
        return scores

    def marginals(self, forest: dict[Atom, list], goal: Atom, weights,
                  semiring=PROB, passes: int = None) -> dict[Rule, Any]:
        """
        The marginal score of each rule in the derivations of a goal:
        the sum over its ground instances of the outside score of the
        head times the weight times the inside scores of the children.

        In the ``PROB`` semiring this equals the weight of the rule times
        the derivative of the inside score of the goal with respect to it.

        Parameters:
            forest : The output of :meth:`seminaive`.
            goal : The atom whose derivations are marginalized.
            weights : Map from :class:`Rule` to weights in the semiring.
            semiring : The :class:`Semiring` in which to evaluate.
            passes : The number of iterations, see :meth:`inside`.
        """
        weights = MappingOrCallable(weights)
        inside_scores = self.inside(forest, weights, semiring, passes)
        outside_scores = self.outside(
            forest, goal, weights, semiring, passes, inside_scores)
        result = {}
        for fact, justifications in forest.items():
            for rule, children in justifications:
                value = semiring.times(
                    outside_scores.get(fact, semiring.zero), weights[rule])
                for child in children:
                    value = semiring.times(
                        value, inside_scores.get(child, semiring.zero))
                result[rule] = semiring.plus(
                    result.get(rule, semiring.zero), value)
        return result

    def viterbi(self, forest: dict[Atom, list], goal: Atom, weights,
                passes: int = None
                ) -> Optional[tuple[float, categorial.Term]]:
        """
        The best derivation of a goal and its score in the ``MAXPLUS``
        semiring, i.e. with weights read as log-probabilities, or ``None``
        if the goal is not derivable.

        Parameters:
            forest : The output of :meth:`seminaive`.
            goal : The atom to derive.
            weights : Map from :class:`Rule` to log-weights.
            passes : The number of iterations, see :meth:`inside`.

        Note
        ----
        As for :meth:`derivations`, only acyclic derivations are decoded;
        the score returned is that of the decoded derivation.
        """
        weights = MappingOrCallable(weights)
        scores = self.inside(forest, weights, MAXPLUS, passes)

        def score(rule, children):
            return weights[rule] + sum(
                scores.get(child, MAXPLUS.zero) for child in children)

        def decode(fact, path):
            if fact in path:
                return None
            for rule, children in sorted(
                    forest.get(fact, []),
                    key=lambda pair: -score(*pair)):
                total, subterms = weights[rule], []
                for child in children:
                    result = decode(child, path | {fact})
                    if result is None:
                        break
                    total, subterms = total + result[0], subterms + [
                        result[1]]
                else:
                    return total, rule(*subterms)
            return None

        return decode(goal, frozenset())

    @classmethod
    def from_tagged(cls, lexicon: cat.Functor, tagging, ob=None
                    ) -> tuple[Program, dict[Rule, float]]:
        """
        Build a program from the output of a supertagger: a sub-lexicon
        restricted to the tagged words together with a weight for each
        rule, e.g. to be used with :meth:`viterbi` or :meth:`inside`.

        Parameters:
            lexicon : The full lexicon from which to take the entries.
            tagging : For each token, an iterable of pairs of a word and
                      its log-weight.
            ob : An optional map from atomic object types to sorts.

        Note
        ----
        When the same word is tagged for several tokens, its rule gets
        the maximum weight: per-token weights on ground rule instances
        are left for future work.
        """
        entries = {}
        for token in tagging:
            for word, weight in token:
                entries[word] = max(
                    entries.get(word, float('-inf')), weight)
        sublexicon = type(lexicon)(
            lexicon.ob_map, {word: lexicon(word) for word in entries})
        program = cls.from_lexicon(sublexicon, ob)
        return program, {
            rule: entries[rule.word] for rule in program.rules}

    def chart(self, term: closed.Term, start: categorial.Ty,
              identify: callable = None) -> tuple[dict[Atom, list], Atom]:
        """
        The packed forest and the goal atom for an input term, i.e. the
        chart from which derivations, scores and marginals are computed.

        Parameters:
            term : The closed term to parse.
            start : The atomic categorial type to derive.
            identify : An optional preprocessing step, see :func:`identify`.
        """
        if self.functor(term.cod) != self.functor(self.lexicon(start)):
            raise ValueError(
                f"Expected a term of type {self.lexicon(start)}, "
                f"got {term.cod}.")
        facts, output = self.database(term, identify)
        return self.seminaive(facts), Atom(start.inside[0].name, output)

    def parse(self, term: closed.Term, start: categorial.Ty,
              identify: callable = None) -> Iterator[categorial.Term]:
        """
        Enumerate the derivations of an input term, i.e. the categorial
        terms whose image under the lexicon beta-reduces to the input.

        Parameters:
            term : The closed term to parse.
            start : The atomic categorial type to derive.
            identify : An optional preprocessing step, see :func:`identify`.
        """
        forest, goal = self.chart(term, start, identify)
        return self.derivations(forest, goal)


def parse(lexicon: cat.Functor, term: closed.Term, start: categorial.Ty,
          identify: callable = None, ob=None) -> Iterator[categorial.Term]:
    """
    Parse a term with a finite lexicon, i.e. enumerate the categorial
    terms whose image under the lexicon beta-reduces to the input.

    Both parsing and generation are instances of the same problem:
    parsing takes a lexicon into strings and an encoded string,
    generation takes a lexicon into logical forms and a logical form.

    Parameters:
        lexicon : A lexicon with a finite mapping from words to terms.
        term : The closed term to parse.
        start : The atomic categorial type to derive.
        identify : An optional preprocessing step, see :func:`identify`.
        ob : An optional map from atomic object types to sorts.
    """
    return Program.from_lexicon(lexicon, ob).parse(term, start, identify)
