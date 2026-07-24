from pytest import raises

from discopy import cat, grammar
from discopy.grammar import categorial
from discopy.grammar.abstract import *
from discopy.utils import AxiomError


def test_Diagram():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.id(x) == Id(x)
    assert Diagram.ev(y, x, left=False) == Eval(x >> y)
    assert Diagram.swap(x, y) == Swap(x, y)
    assert Diagram.copy(x) == Copy(x)
    assert Diagram.discard(x) == Copy(x, 0)
    assert Box('f', x @ y, y).curry() == Curry(Box('f', x @ y, y))

    assert Diagram.fa(x, y) == Eval(x ** y, left=True)
    assert Diagram.ba(x, y) == Eval(x >> y, left=False)
    fc, bc = Diagram.fc(x, y, z), Diagram.bc(x, y, z)
    assert fc.dom == (x ** y) @ (y ** z) and fc.cod == x ** z
    assert bc.dom == (x >> y) @ (y >> z) and bc.cod == x >> z
    assert Diagram.fx(x, y, z) == fc and Diagram.bx(x, y, z) == bc

    left, middle, right = x @ y, y @ z, z @ x
    assert Diagram.fc(left, middle, right).cod == left ** right
    assert Diagram.bc(left, middle, right).cod == left >> right


def test_factory_closure():
    x, y = Ty("x"), Ty("y")
    f, loop = Box("f", x, y), Box("loop", x @ y, x @ y)

    assert type(loop.trace()) is Trace
    assert type(f + f) is Sum
    assert Curry.ob is Sum.ob is Trace.ob is Ty
    assert type(f.to_map()) is CMap and CMap.category is Diagram
    graph = f.to_hypergraph()
    assert type(graph) is Hypergraph
    assert graph.category is Diagram and graph.functor is Functor
    assert type(graph.to_diagram()) is Diagram
    traced = loop.trace().to_hypergraph().to_diagram()
    assert type(traced.boxes[0]) is Trace
    assert issubclass(Lexicon, Functor)


def test_Lexicon_on_diagrams():
    x_, y_, z_ = map(categorial.Ty, "xyz")
    w1, w2 = categorial.Word("w1", y_ << x_), categorial.Word("w2", z_ >> x_)
    diagram = w1 @ w2 >> categorial.Diagram.fx(y_, x_, z_)
    lexicon = Lexicon(
        ob_map=lambda ob: Ty(ob.inside[0].name),
        ar_map=lambda box: Box(box.name, Ty.from_categorial(box.dom),
                               Ty.from_categorial(box.cod)))
    assert lexicon(diagram).cod == Ty("z") >> Ty("y")


def test_Term():
    x, y = Ty('x'), Ty('y')
    f, a = (x >> y)("f"), x("a")
    assert type(f) is Constant
    assert Word("a", x).cod == x and isinstance(Word("a", x), Constant)
    assert type(f(a)) is Application and f(a).cod == y
    var = Variable("v", x)
    assert type(Abstraction(var, f(var))) is Abstraction
    assert Abstraction(var, f(var)).cod == x >> y
    assert eval(str(f(a)), dict(locals())) == f(a)
    assert eval(repr(f(a)), {"grammar": grammar, "cat": cat}) == f(a)
    assert isinstance(f(a).eval(), Diagram)


def test_from_categorial():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    f, g, x = (X >> Y)("f"), (X >> Y)("g"), X("x")
    X_, Y_, Z_ = map(categorial.Ty, "XYZ")
    f_, g_, x_ = (Y_ << X_)("f"), (X_ >> Y_)("g"), X_("x")

    assert type(f_.to_abstract()) is Constant
    assert f_(x_).to_abstract() == f(x)
    assert x_(g_, left=True).to_abstract() == g(x)

    assert categorial.FX(f_, (Z_ >> X_)("g")).to_abstract()\
        == Z(lambda x: f((Z >> X)("g")(x)))
    assert categorial.BX(f_, (Y_ >> Z_)("g")).to_abstract()\
        == X(lambda x: (Y >> Z)("g")(f(x)))
    assert categorial.FC((Z_ << Y_)("h"), f_).to_abstract()\
        == X(lambda x: (Y >> Z)("h")(f(x)))
    assert categorial.FTR(Y_, x_).to_abstract()\
        == (X >> Y)(lambda f: f(x))


def test_open_categorial_terms_preserve_context():
    X, Y, Z = map(categorial.Ty, "XYZ")
    x = categorial.Variable("x", X)
    f_left = categorial.Variable("f", Y << X)
    g_left = categorial.Variable("g", Z << Y)
    f_right = categorial.Variable("f", X >> Y)
    g_right = categorial.Variable("g", Y >> Z)
    terms = [
        categorial.FA(f_left, x),
        categorial.BA(x, f_right),
        categorial.FC(g_left, f_left),
        categorial.BC(f_right, g_right),
        categorial.FX(g_left, f_right),
        categorial.BX(f_left, g_right),
        categorial.FTR(Y, x),
        categorial.BTR(Y, x)]

    for term in terms:
        result = term.to_abstract()
        assert result.dom == Ty.from_categorial(term.dom)
        assert (result.eval().dom, result.eval().cod)\
            == (result.dom, result.cod)

    result = categorial.BX(
        categorial.Variable("x", Y << X), g_right).to_abstract()
    assert result.var.name == "x_"


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
        ob_map={n: ET, np: NP, s: t},
        ar_map={every: EVERY, a: A,
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
        ob_map={n: str, s: bool},
        ar_map={Alice: Function(lambda: "Alice", dom=(), cod=(str, )),
                sleeps: Function(
                    lambda x: x == "Alice",
                    dom=(str, ), cod=(bool, )).curry(left=True)},
        cod=Function)
    assert F(Alice(sleeps, left=True))()
