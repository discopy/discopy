from pytest import raises

from discopy.closed import Ty
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
    assert Diagram.ba(x, y) == BA(x >> y)
    assert Diagram.fa(x, y) == FA(x << y)
    assert Diagram.fc(x, y, z) == FC(x << y, y << z)
    assert Diagram.bc(x, y, z) == BC(x >> y, y >> z)
    assert Diagram.fx(x, y, z) == FX(x << y, z >> y)
    assert Diagram.bx(x, y, z) == BX(y << x, y >> z)


def test_BA():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        BA(x << y)
    assert "BA(closed.Ty(closed.Under(" in repr(BA(x >> y))


def test_FA():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        FA(x >> y)
    assert "FA(closed.Ty(closed.Over" in repr(FA(x << y))


def test_FC():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        FC(x >> y, y >> x)
    with raises(TypeError):
        FC(x << y, y >> x)
    with raises(AxiomError):
        FC(x << y, z << y)


def test_BC():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        BC(x << y, y << x)
    with raises(TypeError):
        BC(x >> y, y << x)
    with raises(AxiomError):
        BC(x >> y, z >> y)


def test_FX():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        FX(x >> y, y >> x)
    with raises(TypeError):
        FX(x << y, y << x)
    with raises(AxiomError):
        FX(x << y, y >> x)


def test_BX():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(TypeError):
        BX(x >> y, y >> x)
    with raises(TypeError):
        BX(x << y, y << x)
    with raises(AxiomError):
        BX(x << y, y >> x)


def test_Functor():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f = Box('f', x, y)
    IdF = Functor(lambda x: x, lambda f: f)
    assert IdF(x >> y << x) == x >> y << x
    assert IdF(Curry(f)) == Curry(f)
    assert IdF(FA(x << y)) == FA(x << y)
    assert IdF(BA(x >> y)) == BA(x >> y)
    assert IdF(FC(x << y, y << x)) == FC(x << y, y << x)
    assert IdF(BC(x >> y, y >> x)) == BC(x >> y, y >> x)
    assert IdF(FX(x << y, z >> y)) == FX(x << y, z >> y)
    assert IdF(BX(y << x, y >> z)) == BX(y << x, y >> z)


def categorial_diagram():
    from discopy.grammar.categorial import Box, Diagram, FA, BA, FC

    S, NP = closed.Ty('S'), closed.Ty('NP')
    boxes = [
        Word('that', NP),
        Word("'s", ((NP >> S) << NP)),
        Word('exactly', ((NP >> S) >> (NP >> S))),
        Box('bx', (((NP >> S) << NP) @ ((NP >> S) >> (NP >> S))),
             ((NP >> S) << NP)),
        Word('what', (NP << (S << NP))),
        Word('i', NP),
        Box('tr', NP, (S << (NP >> S))),
        Word('showed', ((NP >> S) << NP)),
        Word('to', (((NP >> S) >> (NP >> S)) << NP)),
        Word('her', NP),
        FA((((NP >> S) >> (NP >> S)) << NP)),
        Box('bx', (((NP >> S) << NP) @ ((NP >> S) >> (NP >> S))),
             ((NP >> S) << NP)),
        FC((S << (NP >> S)), ((NP >> S) << NP)),
        FA((NP << (S << NP))),
        FA(((NP >> S) << NP)),
        BA((NP >> S)),
    ]
    offsets = [0, 1, 2, 1, 2, 3, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    return Diagram.decode(closed.Ty(), zip(boxes, offsets))


def test_to_tree():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    for diagram in [
            FA(x << y),
            BA(x >> y),
            FC(x << y, y << x),
            BC(x >> y, y >> x),
            FX(x << y, z >> y),
            BX(y << x, y >> z)]:
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
    x, y = closed.Ty('x'), closed.Ty('y')
    x_, y_ = pregroup.Ty('x'), pregroup.Ty('y')
    assert Diagram.to_pregroup(Curry(BA(x >> y))).normal_form()\
        == Cap(y_, y_.l) @ Id(x_)
    assert Diagram.to_pregroup(Curry(FA(x << y), left=False)).normal_form()\
        == Id(y_) @ Cap(x_.r, x_)
    assert Diagram.to_pregroup(FC(x << y, y << x))\
        == Id(x_) @ Cup(y_.l, y_) @ Id(x_.l)
    assert Diagram.to_pregroup(BC(x >> y, y >> x))\
        == Id(x_.r) @ Cup(y_, y_.r) @ Id(x_)
    assert Diagram.to_pregroup(FX(x << y, x >> y))\
        == Id(x_) @ Swap(y_.l, x_.r) @ Id(y_) >>\
        Swap(x_, x_.r) @ Cup(y_.l, y_)
    assert Diagram.to_pregroup(BX(y << x, y >> x))\
        == Id(y_) @ Swap(x_.l, y_.r) @ Id(x_) >>\
        Cup(y_, y_.r) @ Swap(x_.l, x_)


def test_tree2diagram():
    diagram = tree2diagram(tree)
    assert diagram == categorial_diagram()
    assert diagram.to_pregroup() == pregroup_diagram()
