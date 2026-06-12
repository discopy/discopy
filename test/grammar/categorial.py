from pytest import raises

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
    x, f, g = X("x"), (Y << X)("f"), (X >> Y)("g")
    assert (FA(f, x)).cod == Y == BA(x, g).cod


def test_FA():
    x, y = Ty('x'), Ty('y')
    f, a = Constant(x << y, 'f'), Constant(y, 'a')
    with raises(TypeError):
        FA(Constant(x >> y, 'g'), a)
    assert FA(f, a).cod == x


def test_FC():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Constant(x << y, 'f'), Constant(y << z, 'g')
    with raises(TypeError):
        FC(Constant(x >> y, 'f'), Constant(y >> x, 'g'))
    with raises(TypeError):
        FC(f, Constant(y >> x, 'g'))
    with raises(AxiomError):
        FC(f, Constant(z << y, 'g'))
    assert FC(f, g).cod == x << z


def test_BC():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Constant(x >> y, 'f'), Constant(y >> z, 'g')
    with raises(TypeError):
        BC(Constant(x << y, 'f'), Constant(y << x, 'g'))
    with raises(TypeError):
        BC(f, Constant(y << x, 'g'))
    with raises(AxiomError):
        BC(f, Constant(z >> y, 'g'))
    assert BC(f, g).cod == x >> z


def test_FX():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Constant(x << y, 'f'), Constant(z >> y, 'g')
    with raises(TypeError):
        FX(Constant(x >> y, 'f'), Constant(y >> x, 'g'))
    with raises(TypeError):
        FX(f, Constant(y << x, 'g'))
    with raises(AxiomError):
        FX(f, Constant(y >> x, 'g'))
    assert FX(f, g).to_diagram()\
        == Word('f', x << y) @ Word('g', z >> y)\
        >> ForwardCrossedComposition(x << y, z >> y)


def test_BX():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Constant(y << x, 'f'), Constant(y >> z, 'g')
    with raises(TypeError):
        BX(Constant(x >> y, 'f'), Constant(y >> x, 'g'))
    with raises(TypeError):
        BX(f, Constant(y << x, 'g'))
    with raises(AxiomError):
        BX(f, Constant(x >> y, 'g'))
    assert BX(f, g).to_diagram()\
        == Word('f', y << x) @ Word('g', y >> z)\
        >> BackwardCrossedComposition(y << x, y >> z)


def test_Functor():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    IdF = Functor(lambda x: x, lambda f: f)
    assert IdF(x >> y << x) == x >> y << x
    assert IdF(Curry(f)) == Curry(f)
    assert IdF(ForwardCrossedComposition(x << y, z >> y))\
        == ForwardCrossedComposition(x << y, z >> y)
    assert IdF(BackwardCrossedComposition(y << x, y >> z))\
        == BackwardCrossedComposition(y << x, y >> z)


def test_Term():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Constant(x << y, 'f'), Constant(y << z, 'g')
    a, b = Constant(x, 'a'), Constant(x >> y, 'b')
    c, d = Constant(y >> z, 'c'), Constant(z >> y, 'd')
    e, k = Constant(y << x, 'e'), Constant(y >> z, 'k')

    assert isinstance(f, TermBase)
    assert (f << Constant(y, 'arg')).to_diagram()\
        == Word('f', x << y) @ Word('arg', y) >> Eval(x << y)
    assert FA(f, Constant(y, 'arg')).to_diagram()\
        == Word('f', x << y) @ Word('arg', y) >> Eval(x << y)
    assert BA(a, b).to_diagram()\
        == Word('a', x) @ Word('b', x >> y) >> Eval(x >> y)
    assert FX(f, d).cod == z >> x
    assert BX(e, k).cod == z << x
    assert FX(f, d).to_diagram()\
        == Word('f', x << y) @ Word('d', z >> y)\
        >> ForwardCrossedComposition(x << y, z >> y)
    assert BX(e, k).to_diagram()\
        == Word('e', y << x) @ Word('k', y >> z)\
        >> BackwardCrossedComposition(y << x, y >> z)
    assert FC(f, g).simplify() == z(lambda x: f << (g << x))
    assert BC(b, c).simplify() == x(lambda x, left=True: x >> b >> c)


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
