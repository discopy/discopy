import pickle

from discopy import closed
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


def closed_diagram():
    from discopy.closed import Ty, Box, Diagram, FA, BA, FC

    S, NP = Ty('S'), Ty('NP')
    boxes = [
        Word('that', NP),
        Word("'s", ((NP >> S) << NP)),
        Word('exactly', ((NP >> S) >> (NP >> S))),
        Box('bx', (((NP >> S) << NP) @ ((NP >> S) >> (NP >> S))), ((NP >> S) << NP)),
        Word('what', (NP << (S << NP))),
        Word('i', NP),
        Box('tr', NP, (S << (NP >> S))),
        Word('showed', ((NP >> S) << NP)),
        Word('to', (((NP >> S) >> (NP >> S)) << NP)),
        Word('her', NP),
        FA((((NP >> S) >> (NP >> S)) << NP)),
        Box('bx', (((NP >> S) << NP) @ ((NP >> S) >> (NP >> S))), ((NP >> S) << NP)),
        FC((S << (NP >> S)), ((NP >> S) << NP)),
        FA((NP << (S << NP))),
        FA(((NP >> S) << NP)),
        BA((NP >> S)),
    ]
    offsets = [0, 1, 2, 1, 2, 3, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    return Diagram.decode(Ty(), zip(boxes, offsets))


def rigid_diagram():
    from discopy.rigid import Ob, Ty, Box, Cup, Diagram

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


def test_tree2diagram():
    diagram = tree2diagram(tree)
    assert diagram == closed_diagram()
    assert closed.to_rigid(diagram) == rigid_diagram()
