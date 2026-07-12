from pytest import raises

from discopy import closed, frobenius
from discopy.grammar import abstract, categorial
from discopy.grammar.abstract import Ty, Lexicon
from discopy.grammar.datalog import *
from discopy.utils import AxiomError


def montague():
    n, np, s = map(categorial.Ty, ("n", "np", "s"))
    every, a = (np << n)("every"), (np << n)("a")
    woman, man = n("woman"), n("man")
    married = ((np >> s) << np)("married")
    e, t = Ty("e"), Ty("t")
    ET, NP = e >> t, (e >> t) >> t
    forall, exists = (ET >> t)("forall"), (ET >> t)("exists")
    implies, and_ = (t >> (t >> t))("implies"), (t >> (t >> t))("and")
    WOMAN, MAN = ET("WOMAN"), ET("MAN")
    MARRIED = (e >> (e >> t))("MARRIED")
    EVERY = ET(lambda p: ET(lambda q: forall(
        e(lambda x: implies(p(x))(q(x))))))
    A = ET(lambda p: ET(lambda q: exists(
        e(lambda y: and_(p(y))(q(y))))))
    married_sem = NP(lambda o: NP(lambda su: su(
        e(lambda z: o(e(lambda w: MARRIED(w)(z)))))))
    lexicon = Lexicon(
        ob={n: ET, np: NP, s: t},
        ar={every: EVERY, a: A, woman: WOMAN, man: MAN,
            married: married_sem})
    de_dicto = forall(e(lambda x: implies(WOMAN(x))(
        exists(e(lambda y: and_(MAN(y))(MARRIED(y)(x)))))))
    return lexicon, de_dicto, (every, a, woman, man, married), (n, np, s)


def test_Query():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    assert Query.exp(x, y) == Query.over(x, y) == Query.under(x, y) == x @ y
    assert Query.ev(x, y).dom == x @ y @ y and Query.ev(x, y).cod == x
    assert Query.ev(x, y, left=False).dom == y @ x @ y
    assert Query.discard(x) == Query.spiders(1, 0, x)
    f = Query.from_box(frobenius.Box('f', x @ y, x))
    assert f.curry().dom == x and f.curry().cod == x @ y
    assert f.curry() @ Query.id(y) >> Query.ev(x, y) == f
    with raises(NotImplementedError):
        f.curry(left=False)
    assert f.atoms == (Atom('f', (0, 1, 2)), )


def test_Atom():
    assert Atom('f', (0, 1)).substitute({0: 2, 1: 3}) == Atom('f', (2, 3))
    assert Atom('f', (0, )).unify(Atom('g', (1, )), {}) is None
    assert Atom('f', (0, )).unify(Atom('f', (1, 2)), {}) is None
    assert Atom('f', (0, 0)).unify(Atom('f', (1, 2)), {}) is None
    assert Atom('f', (0, 1)).unify(Atom('f', (2, 3)), {}) == {0: 2, 1: 3}


def test_Functor_on_types():
    e, t = Ty("e"), Ty("t")
    F = Functor()
    assert F(e) == frobenius.Ty('e')
    assert F(e >> t) == frobenius.Ty('t', 'e')
    assert F((e >> t) >> t) == frobenius.Ty('t', 't', 'e')
    assert F(e @ t) == frobenius.Ty('e', 't')
    assert F(abstract.Id(e)) == Query.id(frobenius.Ty('e'))


def test_Functor_encode():
    lexicon, _, (every, *_), _ = montague()
    F = Functor()
    EVERY = lexicon(every)
    t, e = frobenius.Ty('t'), frobenius.Ty('e')
    h = F(EVERY)
    assert h.dom == frobenius.Ty() and h.cod == t @ t @ e @ t @ e
    assert h.n_spiders == 5
    assert h.atoms == (
        Atom('forall', (0, 1, 2)), Atom('implies', (1, 3, 4)))

    x = Ty("x")
    var = closed.Variable("v", x)
    assert F.encode(var, (var, )) == Query.id(frobenius.Ty('x'))
    with raises(ValueError):
        F(x(lambda v: (x >> x)("f")(closed.Variable("w", x))))
    with raises(TypeError):
        F.encode(42, ())


def test_almost_linear():
    e, t = Ty("e"), Ty("t")
    ET = e >> t
    and_ = (t >> (t >> t))("and")
    a = e("a")
    with raises(AxiomError):
        Functor()(ET(lambda p: and_(p(a))(p(a))))


def test_Rule_from_word():
    lexicon, _, (every, a, woman, man, married), (n, np, s) = montague()
    program = Program.from_lexicon(lexicon)
    rules = {rule.word.name: rule for rule in program.rules}
    assert rules['every'].head == Atom('np', (0, 3, 2))
    assert rules['every'].children == (Atom('n', (4, 2)), )
    assert rules['every'].constants == (
        Atom('forall', (0, 1, 2)), Atom('implies', (1, 3, 4)))
    assert rules['married'].slots == ((np, False), (np, True))
    assert rules['woman'].body == (Atom('WOMAN', (0, 1)), )

    third_order = (s << (s << np))("that")
    with raises(AxiomError):
        Rule.from_word(third_order, None, lexicon, Functor())
    non_atomic = (n @ n)("pair")
    with raises(AxiomError):
        Rule.from_word(non_atomic, None, lexicon, Functor())


def test_from_lexicon_errors():
    n, s = categorial.Ty("n"), categorial.Ty("s")
    e, t = Ty("e"), Ty("t")
    infinite = Lexicon(ob={n: e}, ar=lambda w: e(w.name))
    with raises(ValueError):
        Program.from_lexicon(infinite)
    clashing = Lexicon(ob={n: e >> t}, ar={n("woman"): (e >> t)("n")})
    with raises(ValueError):
        Program.from_lexicon(clashing)


def test_seminaive():
    n, s = categorial.Ty("n"), categorial.Ty("s")
    e, t = Ty("e"), Ty("t")
    ET = e >> t
    WOMAN = ET("WOMAN")
    # An ambiguous grammar: two words with the same image
    # and a unary cyclic rule n -> n given by the identity.
    woman, lady = n("woman"), n("lady")
    unary = (n >> n)("very")
    lexicon = Lexicon(
        ob={n: ET, s: t},
        ar={woman: WOMAN, lady: WOMAN, unary: ET(lambda p: p)})
    program = Program.from_lexicon(lexicon)
    facts, _ = program.database(WOMAN)
    forest = program.seminaive(facts)
    goal = Atom('n', (0, 1))
    assert len(forest[goal]) == 3    # woman, lady and very(n) itself
    results = list(program.derivations(forest, goal))
    assert results == [woman, lady]  # acyclic derivations only


def test_parse_montague():
    lexicon, de_dicto, (every, a, woman, man, married), (n, np, s) = \
        montague()
    result = list(parse(lexicon, de_dicto, s))
    assert result == [every(woman)(married(a(man)), left=True)]
    assert lexicon(result[0]).normal_form() == de_dicto.normal_form()

    e, t = Ty("e"), Ty("t")
    outside = (e >> (e >> t))("MARRIED")(e("alice"))(e("bob"))
    assert not list(parse(lexicon, outside, s))


def test_parse_string_cfg():
    n, np, s = map(categorial.Ty, ("n", "np", "s"))
    John, unicorn = np("John"), n("unicorn")
    found, a = ((np >> s) << np)("found"), (np << n)("a")
    o = Ty("o")
    STR = o >> o
    john_c, found_c, a_c, unicorn_c = (
        STR(w) for w in ("john", "found", "a", "unicorn"))
    lexicon = Lexicon(
        ob={n: STR, np: STR, s: STR},
        ar={John: o(lambda z: john_c(z)),
            unicorn: o(lambda z: unicorn_c(z)),
            a: STR(lambda x: o(lambda z: a_c(x(z)))),
            found: STR(lambda x: STR(lambda y: o(
                lambda z: y(found_c(x(z))))))})
    sentence = o(lambda z: john_c(found_c(a_c(unicorn_c(z)))))
    program = Program.from_lexicon(lexicon)
    facts, output = program.database(sentence)
    assert facts == {
        Atom('john', (0, 1)), Atom('found', (1, 2)),
        Atom('a', (2, 3)), Atom('unicorn', (3, 4))}
    assert output == (0, 4)
    assert list(program.parse(sentence, s))\
        == [John(found(a(unicorn)), left=True)]
    scrambled = o(lambda z: found_c(john_c(a_c(unicorn_c(z)))))
    assert not list(program.parse(scrambled, s))


def ambiguous():
    n, s = categorial.Ty("n"), categorial.Ty("s")
    e, t = Ty("e"), Ty("t")
    WOMAN = (e >> t)("WOMAN")
    woman, lady = n("woman"), n("lady")
    lexicon = Lexicon(ob={n: e >> t, s: t}, ar={woman: WOMAN, lady: WOMAN})
    program = Program.from_lexicon(lexicon)
    facts, _ = program.database(WOMAN)
    forest = program.seminaive(facts)
    rules = {rule.word.name: rule for rule in program.rules}
    return program, forest, Atom('n', (0, 1)), rules


def test_Semiring():
    assert COUNT.plus(COUNT.one, COUNT.one) == 2
    assert BOOL.times(BOOL.one, BOOL.zero) is False
    assert MAXPLUS.plus(MAXPLUS.zero, 42) == MAXPLUS.times(MAXPLUS.one, 42)
    assert LOGPROB.plus(LOGPROB.zero, 0.5) == 0.5
    assert LOGPROB.plus(0.5, LOGPROB.zero) == 0.5
    from math import exp, log
    assert abs(LOGPROB.plus(log(.3), log(.5)) - log(.8)) < 1e-12


def test_inside():
    program, forest, goal, rules = ambiguous()
    counting = program.inside(
        forest, {rule: 1 for rule in program.rules}, COUNT)
    assert counting[goal] == len(list(program.derivations(forest, goal)))
    weights = {rules['woman']: .3, rules['lady']: .5}
    assert abs(program.inside(forest, weights, PROB)[goal] - .8) < 1e-12
    from math import log
    logweights = {rule: log(w) for rule, w in weights.items()}
    assert abs(program.inside(forest, logweights, LOGPROB)[goal]
               - log(.8)) < 1e-12

    from fractions import Fraction
    exact = Semiring(Fraction(0), Fraction(1),
                     lambda x, y: x + y, lambda x, y: x * y)
    fractions = {rules['woman']: Fraction(3, 10),
                 rules['lady']: Fraction(1, 2)}
    assert program.inside(forest, fractions, exact)[goal] == Fraction(4, 5)


def test_marginals_gradcheck():
    program, forest, goal, rules = ambiguous()
    weights = {rules['woman']: .3, rules['lady']: .5}
    assert program.outside(forest, goal, weights)[goal] == 1.
    marginals = program.marginals(forest, goal, weights, PROB)
    assert abs(marginals[rules['woman']] - .3) < 1e-12
    assert abs(marginals[rules['lady']] - .5) < 1e-12
    eps = 1e-6
    for name in ('woman', 'lady'):
        plus, minus = dict(weights), dict(weights)
        plus[rules[name]] += eps
        minus[rules[name]] -= eps
        finite_difference = (
            program.inside(forest, plus, PROB)[goal]
            - program.inside(forest, minus, PROB)[goal]) / (2 * eps)
        analytic = marginals[rules[name]] / weights[rules[name]]
        assert abs(finite_difference - analytic) < 1e-6


def test_viterbi():
    from math import log
    program, forest, goal, rules = ambiguous()
    logweights = {rules['woman']: log(.3), rules['lady']: log(.5)}
    score, term = program.viterbi(forest, goal, logweights)
    assert term == categorial.Ty("n")("lady")
    assert abs(score - log(.5)) < 1e-12
    assert program.viterbi(forest, Atom('s', (42, )), logweights) is None


def test_weighted_montague():
    from math import log
    lexicon, de_dicto, (every, a, woman, man, married), (n, np, s) = \
        montague()
    program = Program.from_lexicon(lexicon)
    forest, goal = program.chart(de_dicto, s)
    assert program.inside(
        forest, {rule: 1 for rule in program.rules}, COUNT)[goal] == 1
    weights = {rule: .5 for rule in program.rules}
    partition = program.inside(forest, weights, PROB)[goal]
    assert abs(partition - .5 ** 5) < 1e-15
    marginals = program.marginals(forest, goal, weights, PROB)
    assert all(abs(value - partition) < 1e-15
               for value in marginals.values() if value > 0)
    score, best = program.viterbi(
        forest, goal, {rule: log(.5) for rule in program.rules})
    assert best == every(woman)(married(a(man)), left=True)
    assert abs(score - 5 * log(.5)) < 1e-12


def test_weighted_cyclic():
    n, s = categorial.Ty("n"), categorial.Ty("s")
    e, t = Ty("e"), Ty("t")
    ET = e >> t
    WOMAN = ET("WOMAN")
    woman, lady, very = n("woman"), n("lady"), (n >> n)("very")
    lexicon = Lexicon(
        ob={n: ET, s: t},
        ar={woman: WOMAN, lady: WOMAN, very: ET(lambda p: p)})
    program = Program.from_lexicon(lexicon)
    facts, _ = program.database(WOMAN)
    forest = program.seminaive(facts)
    goal = Atom('n', (0, 1))
    rules = {rule.word.name: rule for rule in program.rules}
    assert program.inside(
        forest, {rule: True for rule in program.rules}, BOOL)[goal]
    score, term = program.viterbi(forest, goal, {
        rules['woman']: -.2, rules['lady']: -.1, rules['very']: -1.})
    assert term == lady and abs(score + .1) < 1e-12
    # A profitable cycle is tried first, cut by the guard, then backed off.
    score, term = program.viterbi(forest, goal, {
        rules['woman']: -.2, rules['lady']: -.1, rules['very']: 1.})
    assert term == lady and abs(score + .1) < 1e-12


def test_from_tagged():
    lexicon, de_dicto, (every, a, woman, man, married), (n, np, s) = \
        montague()
    tagging = [
        [(every, -.1), (a, -2.)], [(woman, -.2)], [(married, -.1)],
        [(a, -.3), (every, -1.5)], [(man, -.1)]]
    program, weights = Program.from_tagged(lexicon, tagging)
    assert len(program.rules) == 5
    assert weights[{r.word.name: r for r in program.rules}['every']] == -.1
    forest, goal = program.chart(de_dicto, s)
    _, best = program.viterbi(forest, goal, weights)
    assert best == every(woman)(married(a(man)), left=True)


def test_identify_and_type_mismatch():
    lexicon, de_dicto, _, (n, np, s) = montague()
    program = Program.from_lexicon(lexicon)
    e, t = Ty("e"), Ty("t")
    with raises(ValueError):
        next(program.parse((e >> t)("WOMAN"), s))
    wrong = t("wrong")
    result = list(program.parse(wrong, s, identify=lambda _: de_dicto))
    assert len(result) == 1
