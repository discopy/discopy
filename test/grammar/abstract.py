from random import Random

from pytest import raises

from discopy import cat, grammar
from discopy.grammar import categorial
from discopy.grammar.abstract import *
from discopy.python import Function
from discopy.utils import AxiomError


def tokens(string):
    "The words of a string term, read by interpreting positions as lists."
    python = Functor(
        {Position: list}, lambda word: lambda: lambda xs: xs + [word.name],
        cod=Function)
    return python(string)()([])


def context_free_grammar():
    "Section 4.1 of de Groote (2001): ``S -> epsilon | a S b``."
    S = Ty("S")
    A, B = S("A"), (S >> S)("B")
    lexicon = Lexicon({S: String}, {
        A: Position(lambda x: x),
        B: String(lambda x: String("a").compose(x, String("b")))})
    return A, B, lexicon


def test_factory_closure():
    x, y = Ty("x"), Ty("y")
    f, loop = Box("f", x, y), Box("loop", x @ y, x @ y)
    assert type(Diagram.swap(x, y)) is Swap
    assert type(Diagram.copy(x)) is Copy and type(Diagram.discard(x)) is Discard
    assert type(loop.trace()) is Trace and type(f + f) is Sum
    assert type(f.curry()) is Curry and type(f.curry().uncurry()) is Diagram
    assert all(cls.ob is Ty for cls in (
        Diagram, Box, Eval, Coeval, Curry, Copy, Discard,
        Permutation, Swap, Trace, Sum, Constant, Variable))
    assert type(f.to_map()) is CMap and CMap.category is Diagram
    graph = f.to_hypergraph()
    assert type(graph) is Hypergraph and type(graph.to_diagram()) is Diagram
    assert Diagram.functor_factory is Functor
    assert Lexicon.dom is Lexicon.cod is Diagram


def test_Diagram():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.fa(x, y) == Diagram.ev(x, y) == Eval(y >> x, left=True)
    assert Diagram.ba(x, y) == Eval(x >> y, left=False)
    fc, bc = Diagram.fc(x, y, z), Diagram.bc(x, y, z)
    assert (fc.dom, fc.cod) == ((x << y) @ (y << z), x << z)
    assert (bc.dom, bc.cod) == ((x >> y) @ (y >> z), x >> z)
    assert Diagram.fx(x, y, z) == fc and Diagram.bx(x, y, z) == bc

    left, middle, right = x @ y, y @ z, z @ x
    assert Diagram.fc(left, middle, right).cod == left << right
    assert Diagram.bc(left, middle, right).cod == left >> right


def test_Term():
    x, y = Ty('x'), Ty('y')
    f, a = (x >> y)("f"), x("a")
    assert type(f) is Constant is Word and f.cod == x >> y
    assert type(f(a)) is Application and f(a).cod == y
    assert type(x(lambda v: f(v))) is Abstraction
    assert x(lambda v: f(v)).cod == x >> y and a(f, left=True) == f(a)
    assert eval(str(f(a)), dict(locals())) == f(a)
    assert eval(repr(f(a)), {"grammar": grammar, "cat": cat}) == f(a)
    assert f(a).eval() == f @ a >> Diagram.fa(y, x)


def test_from_categorial():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    f, g, x = (X >> Y)("f"), (X >> Y)("g"), X("x")
    X_, Y_, Z_ = map(categorial.Ty, "XYZ")
    f_, g_, x_ = (Y_ << X_)("f"), (X_ >> Y_)("g"), X_("x")
    assert Ty.from_categorial(Y_ << X_) == Ty.from_categorial(X_ >> Y_)\
        == Ty.from_biclosed(X_ >> Y_) == X >> Y

    assert type(f_.to_abstract()) is Constant
    assert f_(x_).to_abstract() == TermBase.from_biclosed(f_(x_)) == f(x)
    assert x_(g_, left=True).to_abstract() == g(x)
    assert categorial.FX(f_, (Z_ >> X_)("h")).to_abstract()\
        == Z(lambda x: f((Z >> X)("h")(x)))
    assert categorial.BX(f_, (Y_ >> Z_)("h")).to_abstract()\
        == X(lambda x: (Y >> Z)("h")(f(x)))
    assert categorial.FC((Z_ << Y_)("h"), f_).to_abstract()\
        == X(lambda x: (Y >> Z)("h")(f(x)))
    assert categorial.BC(g_, (Y_ >> Z_)("h")).to_abstract()\
        == X(lambda x: (Y >> Z)("h")(g(x)))
    for raised in (categorial.FTR(Y_, x_), categorial.BTR(Y_, x_)):
        assert raised.to_abstract() == (X >> Y)(lambda f: f(x))

    n, s = categorial.Ty("n"), categorial.Ty("s")
    N, S = Ty("n"), Ty("s")
    Alice, loves, Bob = n("Alice"), ((n >> s) << n)("loves"), n("Bob")
    diagram = Alice @ loves @ Bob\
        >> n @ categorial.Diagram.fa(n >> s, n) >> categorial.Diagram.ba(n, s)
    assert Diagram.from_categorial(diagram)\
        == N("Alice") @ (N >> (N >> S))("loves") @ N("Bob")\
        >> N @ Diagram.fa(N >> S, N) >> Diagram.ba(N, S)
    word = categorial.Word("Alice", n, dom=n)
    assert Diagram.from_categorial(word) == Box("Alice", N, N)
    assert Diagram.from_categorial(categorial.Word("Bob", n))\
        == Box("Bob", Ty(), N) != N("Bob")


def test_crossed_composition_requires_symmetry():
    x, y, z = map(categorial.Ty, "xyz")
    fx = categorial.FX((y << x)("f"), (z >> x)("g"))
    bx = categorial.BX((y << x)("f"), (y >> z)("g"))
    F = categorial.Functor(
        ob_map=lambda ob: categorial.Ty(ob.inside[0].name),
        ar_map=lambda c: categorial.Constant(c.name, c.cod),
        cod=categorial.Diagram)
    with raises(AxiomError):
        F(fx)
    with raises(AxiomError):
        F(bx)


def test_Lexicon():
    x, y, z = map(Ty, "xyz")
    a, b, c = x("a"), y("b"), z("c")
    first = Lexicon(ob_map={x: y}, ar_map={a: b})
    second = Lexicon(ob_map={y: z}, ar_map={b: c})
    assert first(a) == b and first(x >> x) == y >> y
    assert first(x(lambda v: v)) == y(lambda v: v)
    assert (first >> second)(a) == second(first(a)) == c
    assert type(first >> second) is Lexicon
    with raises(AxiomError):
        Lexicon(ob_map={x: y}, ar_map={a: (y >> y)("b")})(a)
    for outside in (z, c, z(lambda v: v)):
        with raises(KeyError):
            first(outside)


def test_strings():
    Alice, loves, Bob = map(String, ("Alice", "loves", "Bob"))
    empty = Position(lambda x: x)
    assert String == Position >> Position
    sentence = Alice.compose(loves, Bob)
    assert sentence.is_linear
    assert sentence == Position(lambda x: Bob(loves(Alice(x))))
    assert tokens(sentence) == tokens(Alice.compose(loves.compose(Bob)))\
        == ["Alice", "loves", "Bob"]
    assert tokens(empty.compose(Alice)) == tokens(Alice.compose(empty))\
        == ["Alice"]


def test_categorial_lexicon():
    "The string lexicon of a categorial grammar, written per word."
    n, s = categorial.Ty("n"), categorial.Ty("s")
    Alice, loves, Bob, sleeps = (
        n("Alice"), ((n >> s) << n)("loves"), n("Bob"), (n >> s)("sleeps"))
    LOVES, SLEEPS = String("loves"), String("sleeps")
    strings = categorial.Functor(
        ob_map={n: String, s: String},
        ar_map={Alice: String("Alice"), Bob: String("Bob"),
                loves: String(lambda o: String(lambda x: x.compose(LOVES, o))),
                sleeps: String(lambda x: x.compose(SLEEPS))},
        cod=Diagram)
    assert strings(n >> s) == strings(s << n) == String >> String
    sentences = [
        Alice(loves(Bob), left=True),
        categorial.FC(categorial.FTR(s, Alice), loves)(Bob),
        Alice(loves(categorial.BTR(n >> s, Bob), left=True), left=True),
        Alice @ loves @ Bob >> n @ categorial.Diagram.fa(n >> s, n)
        >> categorial.Diagram.ba(n, s)]
    for sentence in sentences:
        assert tokens(strings(sentence)) == ["Alice", "loves", "Bob"]
    raised = strings(categorial.FTR(s, Alice)(sleeps))
    assert tokens(raised) == ["Alice", "sleeps"]

    w, g = (s << n)("w"), (n >> n)("g")
    crossed = categorial.Functor({n: String, s: String}, {
        Alice: String("Alice"), w: String(lambda x: String("w").compose(x)),
        g: String(lambda x: x.compose(String("g")))}, cod=Diagram)
    assert tokens(crossed(Alice(categorial.FX(w, g), left=True)))\
        == ["w", "Alice", "g"]


def test_Montague_semantics():
    """
    Two sentences with the same grammatical structure and different
    quantifier scopes, checked against Python over a random finite universe
    as in the higher-order DisCoCat notebook: "Every woman married a man" is
    read de dicto, i.e. for every woman there is a man that she married, and
    "Every child learnt a song" de re, i.e. there is a song that every child
    learnt. The semantic terms copy their variables of ground type: they
    normalise to the two first-order formulas and evaluate all the same.
    """
    n, np, s = map(categorial.Ty, ("n", "np", "s"))
    every, a = (np << n)("every"), (np << n)("a")
    woman, man, child, song = (n(w) for w in ("woman", "man", "child", "song"))
    married, learnt = (((np >> s) << np)(v) for v in ("married", "learnt"))

    def sentence(det1, noun1, verb, det2, noun2):
        return det1(noun1)(verb(det2(noun2)), left=True)

    every_woman_married_a_man = sentence(every, woman, married, a, man)
    every_child_learnt_a_song = sentence(every, child, learnt, a, song)

    e, t = Ty("e"), Ty("t")
    Predicate, Quantifier = e >> t, (e >> t) >> t
    forall, exists = Quantifier("forall"), Quantifier("exists")
    implies, and_ = (t >> (t >> t))("implies"), (t >> (t >> t))("and")
    WOMAN, MAN, CHILD, SONG = (
        Predicate(w) for w in ("WOMAN", "MAN", "CHILD", "SONG"))
    MARRIED, LEARNT = ((e >> (e >> t))(v) for v in ("MARRIED", "LEARNT"))
    EVERY = Predicate(lambda p: Predicate(lambda q: forall(
        e(lambda x: implies(p(x))(q(x))))))
    A = Predicate(lambda p: Predicate(lambda q: exists(
        e(lambda y: and_(p(y))(q(y))))))
    de_dicto = Quantifier(lambda o: Quantifier(lambda su: su(
        e(lambda x: o(e(lambda y: MARRIED(x)(y)))))))
    de_re = Quantifier(lambda o: Quantifier(lambda su: o(
        e(lambda y: su(e(lambda x: LEARNT(x)(y)))))))
    semantics = categorial.Functor(
        ob_map={n: Predicate, np: Quantifier, s: t}, cod=Diagram,
        ar_map=lambda word: {
            "every": EVERY, "a": A, "woman": WOMAN, "man": MAN,
            "child": CHILD, "song": SONG,
            "married": de_dicto, "learnt": de_re}[word.name])
    for term in (every_woman_married_a_man, every_child_learnt_a_song):
        formula = semantics(term)
        assert formula.cod == t and not formula.is_linear
        diagram = formula.eval()
        assert (diagram.dom, diagram.cod) == (Ty(), t)
        assert not diagram.is_linear

    random, U = Random(42), range(8)
    Woman, Man, Child, Song = (
        [random.choice([True, False]) for _ in U] for _ in range(4))
    Married, Learnt = (
        {(x, y): random.choice([True, False]) for x in U for y in U}
        for _ in range(2))
    python = Functor(ob_map={e: int, t: bool}, ar_map={
        forall: lambda: lambda p: all(p(x) for x in U),
        exists: lambda: lambda p: any(p(x) for x in U),
        implies: lambda: lambda p: lambda q: not p or q,
        and_: lambda: lambda p: lambda q: p and q,
        WOMAN: lambda: lambda x: Woman[x], MAN: lambda: lambda x: Man[x],
        CHILD: lambda: lambda x: Child[x], SONG: lambda: lambda x: Song[x],
        MARRIED: lambda: lambda x: lambda y: Married[x, y],
        LEARNT: lambda: lambda x: lambda y: Learnt[x, y]}, cod=Function)
    assert python(semantics(every_woman_married_a_man))() == all(
        not Woman[x] or any(Man[y] and Married[x, y] for y in U) for x in U)
    assert python(semantics(every_child_learnt_a_song))() == any(
        Song[y] and all(not Child[x] or Learnt[x, y] for x in U) for y in U)


def test_context_free_grammar():
    term, B, lexicon = context_free_grammar()
    for n in range(5):
        assert term.cod == Ty("S") and not term.freevars and term.is_linear
        image = lexicon(term)
        assert image.cod == String and not image.freevars and image.is_linear
        assert tokens(image) == n * ["a"] + n * ["b"]
        term = B(term)


def test_tree_adjoining_grammar():
    "Section 4.3 of de Groote (2001): adjunction generates a^n b^n c^n d^n."
    S, Sp, Spp = map(Ty, ("S", "S'", "S''"))
    auxiliary = Spp >> Sp
    A, B, C = ((auxiliary >> S)("A"),
               (Spp >> (auxiliary >> Sp))("B"), auxiliary("C"))
    a, b, c, d = map(String, "abcd")
    lexicon = Lexicon({S: String, Sp: String, Spp: String}, {
        A: (String >> String)(lambda f: f(Position(lambda x: x))),
        B: String(lambda x: (String >> String)(
            lambda g: a.compose(g(b.compose(x, c)), d))),
        C: String(lambda x: x)})
    adjunct = C
    for n in range(5):
        term = A(adjunct)
        assert term.cod == S and not term.freevars and term.is_linear
        image = lexicon(term)
        assert image.cod == String and not image.freevars and image.is_linear
        assert tokens(image) == [letter for letter in "abcd" for _ in range(n)]
        adjunct = Spp(lambda x: B(x)(adjunct))


def test_Lexicon_composition():
    "Section 6 of de Groote (2001): a second lexicon reads the object words."
    A, B, first = context_free_grammar()
    second = Lexicon({Position: Position}, {
        String("a"): String("A").compose(String("A")),
        String("b"): String("B")})
    term = B(B(A))
    sequential, composed = second(first(term)), (first >> second)(term)
    assert tokens(sequential) == tokens(composed) == list("AAAABB")


def test_ambiguity():
    "Two abstract terms with one string and two meanings."
    n, s, e, t = map(Ty, "nset")
    river, finance = n("river bank"), n("financial bank")
    visit = (n >> s)("visit")
    syntax = Lexicon({n: String, s: String}, {
        river: String("bank"), finance: String("bank"),
        visit: String(lambda x: String("visit").compose(x))})
    RIVER, FINANCE, VISIT = e("RIVER"), e("FINANCE"), (e >> t)("VISIT")
    semantics = Lexicon({n: e, s: t}, {
        river: RIVER, finance: FINANCE, visit: VISIT})
    readings = visit(river), visit(finance)
    forms = [syntax(reading) for reading in readings]
    assert readings[0] != readings[1]
    assert tokens(forms[0]) == tokens(forms[1]) == ["visit", "bank"]
    assert [semantics(reading) for reading in readings]\
        == [VISIT(RIVER), VISIT(FINANCE)]
