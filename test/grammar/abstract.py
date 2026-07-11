from pytest import raises

from discopy import cat, grammar
from discopy.grammar import categorial
from discopy.grammar.abstract import *
from discopy.utils import AxiomError


def test_Diagram():
    x, y = Ty('x'), Ty('y')
    assert Diagram.id(x) == Id(x)
    assert Diagram.ev(y, x, left=False) == Eval(x >> y)
    assert Diagram.swap(x, y) == Swap(x, y)
    assert Diagram.copy(x) == Copy(x)
    assert Diagram.discard(x) == Copy(x, 0)
    assert Box('f', x @ y, y).curry() == Curry(Box('f', x @ y, y))


def test_Term():
    x, y = Ty('x'), Ty('y')
    f, a = (x >> y)("f"), x("a")
    assert isinstance(f, Constant)
    assert f(a).cod == y
    var = Variable("v", x)
    assert Abstraction(var, f(var)).cod == x >> y
    assert eval(str(f(a)), dict(locals())) == f(a)
    assert eval(repr(f(a)), {"grammar": grammar, "cat": cat}) == f(a)
    assert isinstance(f(a).eval(), Diagram)


def test_from_categorial():
    X, Y = Ty("X"), Ty("Y")
    f, g, x = (X >> Y)("f"), (X >> Y)("g"), X("x")
    X_, Y_, Z_ = map(categorial.Ty, "XYZ")
    f_, g_, x_ = (Y_ << X_)("f"), (X_ >> Y_)("g"), X_("x")

    assert f_(x_).to_abstract() == f(x)
    assert x_(g_, left=True).to_abstract() == g(x)

    fx = categorial.FX((y << x)("f"), (z >> x)("g"))
    term = TermBase.from_categorial(fx)
    assert isinstance(term, Abstraction) and term.cod == Z >> Y

    bx = categorial.BX((y << x)("f"), (y >> z)("g"))
    term = TermBase.from_categorial(bx)
    assert isinstance(term, Abstraction) and term.cod == X >> Z

    fc = categorial.FC((z << y)("f"), (y << x)("g"))
    term = TermBase.from_categorial(fc)
    assert isinstance(term, Abstraction) and term.cod == X >> Z

    ftr = categorial.FTR(y, a)
    term = TermBase.from_categorial(ftr)
    assert isinstance(term, Abstraction) and term.cod == (X >> Y) >> Y


def test_crossed_composition_requires_symmetry():
    x, y, z = map(categorial.Ty, "xyz")
    fx = categorial.FX((y << x)("f"), (z >> x)("g"))
    F = categorial.Functor(
        ob=lambda ob: categorial.Ty(ob.name),
        ar=lambda c: categorial.Constant(c.name, c.cod),
        cod=categorial.Diagram)
    with raises(AxiomError):
        F(fx)


def test_Lexicon_Montague_semantics():
    # Syntax: two sentences with the same grammatical structure.
    n, np, s = map(categorial.Ty, ("n", "np", "s"))
    every, a = (np << n)("every"), (np << n)("a")
    woman, man, child, song = (n(w) for w in ("woman", "man", "child", "song"))
    married, learnt = (((np >> s) << np)(v) for v in ("married", "learnt"))

    def sentence(det1, noun1, verb, det2, noun2):
        return det1(noun1)(verb(det2(noun2)), left=True)

    every_woman_married_a_man = sentence(every, woman, married, a, man)
    every_child_learnt_a_song = sentence(every, child, learnt, a, song)
    assert every_woman_married_a_man.cod == every_child_learnt_a_song.cod == s

    # Semantics: logical formulas as lambda terms with higher-order
    # constants for the quantifiers and connectives.
    e, t = Ty("e"), Ty("t")
    ET, NP = e >> t, (e >> t) >> t
    forall, exists = (ET >> t)("forall"), (ET >> t)("exists")
    implies, and_ = (t >> (t >> t))("implies"), (t >> (t >> t))("and")
    WOMAN, MAN, CHILD, SONG = (
        ET(w) for w in ("WOMAN", "MAN", "CHILD", "SONG"))
    MARRIED, LEARNT = ((e >> (e >> t))(v) for v in ("MARRIED", "LEARNT"))

    EVERY = ET(lambda p: ET(lambda q: forall(
        e(lambda x: implies(p(x))(q(x))))))
    A = ET(lambda p: ET(lambda q: exists(
        e(lambda y: and_(p(y))(q(y))))))

    # "married" is interpreted de dicto, "learnt" de re: the object
    # quantifier scopes below or above the subject quantifier.
    married_sem = NP(lambda o: NP(lambda su: su(
        e(lambda z: o(e(lambda w: MARRIED(w)(z)))))))
    learnt_sem = NP(lambda o: NP(lambda su: o(
        e(lambda w: su(e(lambda z: LEARNT(w)(z)))))))

    lexicon = Lexicon(
        ob={n: ET, np: NP, s: t},
        ar={every: EVERY, a: A,
            woman: WOMAN, man: MAN, child: CHILD, song: SONG,
            married: married_sem, learnt: learnt_sem})

    # For every woman there exists a man that she married.
    de_dicto = forall(e(lambda x: implies(WOMAN(x))(
        exists(e(lambda y: and_(MAN(y))(MARRIED(y)(x)))))))
    # There exists a song that every child learnt.
    de_re = exists(e(lambda y: and_(SONG(y))(
        forall(e(lambda x: implies(CHILD(x))(LEARNT(y)(x)))))))

    assert lexicon(every_woman_married_a_man).normal_form() == de_dicto
    assert lexicon(every_child_learnt_a_song).normal_form() == de_re


def test_python_Functor_on_terms():
    from discopy.python import Function
    n, s = categorial.Ty("n"), categorial.Ty("s")
    Alice, sleeps = n("Alice"), (n >> s)("sleeps")
    F = categorial.Functor(
        ob={n: str, s: bool},
        ar={Alice: Function(lambda: "Alice", dom=(), cod=(str, )),
            sleeps: Function(
                lambda x: x == "Alice",
                dom=(str, ), cod=(bool, )).curry(left=True)},
        cod=Function)
    assert F(Alice(sleeps, left=True))()
