from pytest import raises

from discopy import cat, grammar, biclosed
from discopy.utils import from_tree
from discopy.biclosed import Ty
from discopy.grammar.categorial import *

tree = {
  'type': 'ba',
  'cat': 'S[dcl]',
  'children': [{'word': 'that',
    'lemma': 'XX',
    'pos': 'XX',
    'entity': 'XX',
    'chunk': 'XX',
    'cat': 'NP'},
   {'type': 'fa',
    'cat': 'S[dcl]\\NP',
    'children': [{'type': 'bx',
      'cat': '(S[dcl]\\NP)/NP',
      'children': [{'word': "'s",
        'lemma': 'XX',
        'pos': 'XX',
        'entity': 'XX',
        'chunk': 'XX',
        'cat': '(S[dcl]\\NP)/NP'},
       {'word': 'exactly',
        'lemma': 'XX',
        'pos': 'XX',
        'entity': 'XX',
        'chunk': 'XX',
        'cat': '(S\\NP)\\(S\\NP)'}]},
     {'type': 'fa',
      'cat': 'NP',
      'children': [{'word': 'what',
        'lemma': 'XX',
        'pos': 'XX',
        'entity': 'XX',
        'chunk': 'XX',
        'cat': 'NP/(S[dcl]/NP)'},
       {'type': 'fc',
        'cat': 'S[dcl]/NP',
        'children': [{'type': 'tr',
          'cat': 'S[X]/(S[X]\\NP)',
          'children': [{'word': 'i',
            'lemma': 'XX',
            'pos': 'XX',
            'entity': 'XX',
            'chunk': 'XX',
            'cat': 'NP'}]},
         {'type': 'bx',
          'cat': '(S[dcl]\\NP)/NP',
          'children': [{'word': 'showed',
            'lemma': 'XX',
            'pos': 'XX',
            'entity': 'XX',
            'chunk': 'XX',
            'cat': '(S[dcl]\\NP)/NP'},
           {'type': 'fa',
            'cat': '(S\\NP)\\(S\\NP)',
            'children': [{'word': 'to',
              'lemma': 'XX',
              'pos': 'XX',
              'entity': 'XX',
              'chunk': 'XX',
              'cat': '((S\\NP)\\(S\\NP))/NP'},
             {'word': 'her',
              'lemma': 'XX',
              'pos': 'XX',
              'entity': 'XX',
              'chunk': 'XX',
              'cat': 'NP'}]}]}]}]}]}]}


def test_Diagram():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert Diagram.id(x) == Id(x)
    assert Diagram.ba(x, y) == Eval(x >> y)
    assert Diagram.fa(x, y) == Eval(x << y)
    assert Diagram.fc(x, y, z)\
        == (Id(x << y) @ Id(y << z) @ Id(z)
            >> Id(x << y) @ Eval(y << z)
            >> Eval(x << y)).curry(left=True)
    assert Diagram.bc(x, y, z)\
        == (Id(x) @ Id(x >> y) @ Id(y >> z)
            >> Eval(x >> y) @ Id(y >> z)
            >> Eval(y >> z)).curry()
    assert Diagram.fx(x, y, z) == ForwardCrossedComposition(x << y, z >> y)
    assert Diagram.bx(x, y, z) == BackwardCrossedComposition(y << x, y >> z)


def test_BA_FA():
    X, Y = Ty('X'), Ty('Y')
    x = Constant("x", X)
    f, g = Constant("f", Y << X), Constant("g", X >> Y)

    assert FA(f, x).cod == Y == BA(x, g).cod

    with raises(TypeError):
        FA(x, f)
    with raises(TypeError):
        FA(g, x)

    assert FA(f, x).eval() == f.eval() @ x.eval() >> Eval(Y << X)


def test_terms_simplify_and_eval():
    X, Y, Z = Ty('X'), Ty('Y'), Ty('Z')
    x = Constant("x", X)
    f_left = Constant("f", Y << X)
    g_left = Constant("g", Z << Y)
    f_right = Constant("f", X >> Y)
    g_right = Constant("g", Y >> Z)
    var = Variable("x", X)
    abstraction = Abstraction(var, f_left(var))

    assert x.simplify() is x
    assert var.simplify() is var
    assert abstraction.simplify() == abstraction
    assert f_left(x).simplify() == f_left(x)
    assert x(f_right, left=True).simplify() == x(f_right, left=True)

    for term in [FTR(Y, x), BTR(Y, x)]:
        assert term.freevars == []
        assert str(term) == f"{type(term).__name__}({Y}, {x})"
        assert isinstance(term.simplify(), Abstraction)
        assert term.eval() == term.simplify().eval()
    assert FTR(Y, x).cod == Y << (X >> Y)
    assert BTR(Y, x).cod == (Y << X) >> Y

    fc, bc = FC(g_left, f_left), BC(f_right, g_right)
    assert fc.freevars == []
    assert str(fc) == f"FC({g_left}, {f_left})"
    x_var = Variable("x", X)
    assert fc.simplify() == Abstraction(x_var, g_left(f_left(x_var)))
    assert bc.simplify() == Abstraction(
        x_var, x_var(f_right, left=True)(g_right, left=True), left=True)
    assert fc.eval() == fc.simplify().eval()
    assert bc.eval() == bc.simplify().eval()

    assert FX(g_left, f_right).simplify() == FX(g_left, f_right)
    assert BX(f_left, g_right).simplify() == BX(f_left, g_right)


def test_complex_sentence():
    n, p, s = Ty('n'), Ty('p'), Ty('s')

    That = Constant("That", n)
    was = Constant("was", (n >> s) << n)
    exactly = Constant("exactly", n << n)
    what = Constant("what", n << (s << n))
    I = Constant("I", n)
    showed = Constant("showed", ((n >> s) << p) << n)
    to = Constant("to", p << n)
    her = Constant("her", n)

    sentence = That(was(exactly(what(FC(
        FTR(s, I), BX(showed, BTR((n >> s), to(her))))))), left=True)

    assert sentence.constants == [That, was, exactly, what, I, showed, to, her]
    assert eval(str(sentence)) == eval(repr(sentence)) == sentence


def test_FC_BC_FX_BX():
    X, Y, Z = Ty('X'), Ty('Y'), Ty('Z')
    f_left, g_left = Constant("f", Y << X), Constant("g", Z << Y)
    f_right, g_right = Constant("f", X >> Y), Constant("g", Y >> Z)

    assert FC(g_left, f_left).cod == BX(f_left, g_right).cod == Z << X
    assert BC(f_right, g_right).cod == FX(g_left, f_right).cod == X >> Z

    with raises(TypeError):
        FC(f_right, g_left)
    with raises(TypeError):
        BC(g_left, f_right)
    with raises(TypeError):
        FX(f_right, g_right)
    with raises(TypeError):
        BX(g_right, f_right)

    with raises(AxiomError):
        FC(f_left, g_left)
    with raises(AxiomError):
        BC(g_right, f_right)
    with raises(AxiomError):
        FX(f_left, g_right)
    with raises(AxiomError):
        BX(g_left, f_right)

    repeated = Variable("repeated", X << X)
    with raises(ValueError):
        FC(repeated, repeated)
    with raises(AxiomError):
        ForwardCrossedComposition(X << Y, Z >> X)
    with raises(AxiomError):
        BackwardCrossedComposition(Y << X, Z >> X)

    assert FX(g_left, f_right).eval()\
        == g_left @ f_right\
        >> ForwardCrossedComposition(Z << Y, X >> Y)
    assert BX(f_left, g_right).eval()\
        == f_left @ g_right\
        >> BackwardCrossedComposition(Y << X, Y >> Z)


def test_Functor():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    F = Functor(lambda x: x, lambda f: f)
    assert F(x >> y << x) == x >> y << x
    assert F(Curry(f)) == Curry(f)
    assert F(ForwardCrossedComposition(x << y, z >> y))\
        == ForwardCrossedComposition(x << y, z >> y)
    assert F(BackwardCrossedComposition(y << x, y >> z))\
        == BackwardCrossedComposition(y << x, y >> z)


def test_to_tree():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    for diagram in [
            ForwardCrossedComposition(x << y, z >> y),
            BackwardCrossedComposition(y << x, y >> z)]:
        assert from_tree(diagram.to_tree()) == diagram


def pregroup_diagram():
    from discopy.grammar.pregroup import Ty, Box, Cup, Diagram
    from discopy.rigid import Ob

    boxes = [
        Box('that', Ty(), Ty('NP')),
        Box("'s", Ty(), Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1))),
        Box('exactly', Ty(), Ty(Ob('S', z=1), Ob('NP', z=2), Ob('NP', z=1), 'S')),
        Box('bx', Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1), Ob('S', z=1), Ob('NP', z=2), Ob('NP', z=1), 'S'), Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1))),
        Box('what', Ty(), Ty('NP', Ob('NP', z=-2), Ob('S', z=-1))),
        Box('i', Ty(), Ty('NP')),
        Box('tr', Ty('NP'), Ty('S', Ob('S', z=-1), 'NP')),
        Box('showed', Ty(), Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1))),
        Box('to', Ty(), Ty(Ob('S', z=1), Ob('NP', z=2), Ob('NP', z=1), 'S', Ob('NP', z=-1))),
        Box('her', Ty(), Ty('NP')),
        Cup(Ty(Ob('NP', z=-1)), Ty('NP')),
        Box('bx', Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1), Ob('S', z=1), Ob('NP', z=2), Ob('NP', z=1), 'S'), Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1))),
        Cup(Ty('NP'), Ty(Ob('NP', z=1))),
        Cup(Ty(Ob('S', z=-1)), Ty('S')),
        Cup(Ty(Ob('S', z=-1)), Ty('S')),
        Cup(Ty(Ob('NP', z=-2)), Ty(Ob('NP', z=-1))),
        Cup(Ty(Ob('NP', z=-1)), Ty('NP')),
        Cup(Ty('NP'), Ty(Ob('NP', z=1))),
    ]
    offsets = [0, 1, 4, 1, 4, 7, 7, 10, 13, 18, 17, 10, 9, 8, 6, 5, 3, 0]
    return Diagram.decode(Ty(), zip(boxes, offsets))


def test_to_pregroup():
    from discopy.grammar import pregroup
    from discopy.grammar.pregroup import Cup, Cap, Id, Swap
    x, y = biclosed.Ty('x'), biclosed.Ty('y')
    x_, y_ = pregroup.Ty('x'), pregroup.Ty('y')
    assert Diagram.to_pregroup(Diagram.ba(x, y).curry(left=True))\
        .normal_form()\
        == Cap(y_, y_.l) @ Id(x_)
    assert Diagram.to_pregroup(Diagram.fa(x, y).curry())\
        .normal_form()\
        == Id(y_) @ Cap(x_.r, x_)
    assert Diagram.to_pregroup(Diagram.fc(x, y, x)).normal_form()\
        == Id(x_) @ Cup(y_.l, y_) @ Id(x_.l)
    assert Diagram.to_pregroup(Diagram.bc(x, y, x)).normal_form()\
        == Id(x_.r) @ Cup(y_, y_.r) @ Id(x_)
    assert Diagram.to_pregroup(ForwardCrossedComposition(x << y, x >> y))\
        == Id(x_) @ Swap(y_.l, x_.r) @ Id(y_) >>\
        Swap(x_, x_.r) @ Cup(y_.l, y_)
    assert Diagram.to_pregroup(BackwardCrossedComposition(y << x, y >> x))\
        == Id(y_) @ Swap(x_.l, y_.r) @ Id(x_) >>\
        Cup(y_, y_.r) @ Swap(x_.l, x_)


def test_tree2diagram():
    diagram = tree2diagram(tree)
    assert diagram.to_pregroup().normal_form()\
        == pregroup_diagram().normal_form()

    custom_tree = {
        "type": "custom",
        "cat": "S",
        "children": [
            {"word": "word", "cat": "NP"}]}
    assert tree2diagram(custom_tree) == Word("word", Ty("NP"))\
        >> Box("custom", Ty("NP"), Ty("S"))
    assert tree2diagram({"word": "word", "cat": "NP"}, dom=Ty("D"))\
        == Word("word", Ty("NP"), dom=Ty("D"))
