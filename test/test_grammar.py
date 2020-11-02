import os
from pytest import raises
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from discopy import rigid, messages
from discopy.rigid import Id, Cup, Cap, Ty, Box
from discopy.grammar import *


def test_Word():
    with raises(TypeError):
        Word(0, Ty('n'))
    with raises(TypeError):
        Word('Alice', 0)
    with raises(TypeError):
        Word('Alice', Ty('n'), dom=0)


def test_CFG():
    s, n, v, vp = Ty('S'), Ty('N'), Ty('V'), Ty('VP')
    R0, R1 = Box('R0', vp @ n, s), Box('R1', n @ v, vp)
    Jane, loves, Bob = Word('Jane', n), Word('loves', v), Word('Bob', n)
    cfg = CFG(R0, R1, Jane, loves, Bob)
    assert Jane in cfg.productions
    assert "CFG(Box('R0', Ty('VP', 'N'), Ty('S'))" in repr(cfg)
    assert not list(CFG().generate(start=s, max_sentences=1, max_depth=1))
    sentence, *_ = cfg.generate(
        start=s, max_sentences=1, max_depth=10, not_twice=[Jane, Bob], seed=42)
    assert sentence\
        == (Jane @ loves @ Bob).normal_form(left=True) >> R1 @ Id(n) >> R0


def test_eager_parse():
    s, n = Ty('s'), Ty('n')
    Alice = Word('Alice', n)
    loves = Word('loves', n.r @ s @ n.l)
    Bob = Word('Bob', n)
    grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    assert eager_parse(Alice, loves, Bob) == grammar << Alice @ loves @ Bob
    who = Word('who', n.r @ n @ s.l @ n)
    assert eager_parse(Bob, who, loves, Alice, target=n).offsets ==\
        [0, 1, 5, 8, 0, 2, 1, 1]
    with raises(NotImplementedError):
        eager_parse(Alice, Bob, loves)
    with raises(NotImplementedError):
        eager_parse(Alice, loves, Bob, who, loves, Alice)


def test_brute_force():
    s, n = Ty('s'), Ty('n')
    Alice = Word('Alice', n)
    loves = Word('loves', n.r @ s @ n.l)
    Bob = Word('Bob', n)
    grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    gen = brute_force(Alice, loves, Bob)
    assert next(gen) == Alice @ loves @ Alice >> grammar
    assert next(gen) == Alice @ loves @ Bob >> grammar
    assert next(gen) == Bob @ loves @ Alice >> grammar
    assert next(gen) == Bob @ loves @ Bob >> grammar
    gen = brute_force(Alice, loves, Bob, target=n)
    assert next(gen) == Word('Alice', Ty('n'))
    assert next(gen) == Word('Bob', Ty('n'))


def test_pregroup_draw_errors():
    n = Ty('n')
    with raises(TypeError):
        draw(0)
    with raises(ValueError) as err:
        draw(Cap(n, n.l))
    with raises(ValueError) as err:
        draw(Cup(n, n.r))
    with raises(ValueError) as err:
        draw(Word('Alice', n) >> Word('Alice', n) @ Id(n))
    assert str(err.value) is messages.expected_pregroup()


def test_tree2diagram():
    tree = {'type': 'ba',
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
    diagram = tree2diagram(tree)
    from discopy.biclosed import (
        Ty, Over, Under, Box, FA, BA, Functor, biclosed2rigid)
    from discopy.ccg import Word
    assert diagram.boxes == [
        Word('that', Ty('NP')),
        Word("'s", Over(Under(Ty('NP'), Ty('S')), Ty('NP'))),
        Word('exactly', Under(Under(Ty('NP'), Ty('S')), Under(Ty('NP'), Ty('S')))),
        Box('bx', Ty(Over(Under(Ty('NP'), Ty('S')), Ty('NP')), Under(Under(Ty('NP'), Ty('S')), Under(Ty('NP'), Ty('S')))), Over(Under(Ty('NP'), Ty('S')), Ty('NP'))),
        Word('what', Over(Ty('NP'), Over(Ty('S'), Ty('NP')))),
        Word('i', Ty('NP')),
        Box('tr', Ty('NP'), Over(Ty('S'), Under(Ty('NP'), Ty('S')))),
        Word('showed', Over(Under(Ty('NP'), Ty('S')), Ty('NP'))),
        Word('to', Over(Under(Under(Ty('NP'), Ty('S')), Under(Ty('NP'), Ty('S'))), Ty('NP'))),
        Word('her', Ty('NP')),
        FA(Over(Under(Under(Ty('NP'), Ty('S')), Under(Ty('NP'), Ty('S'))), Ty('NP'))),
        Box('bx', Ty(Over(Under(Ty('NP'), Ty('S')), Ty('NP')), Under(Under(Ty('NP'), Ty('S')), Under(Ty('NP'), Ty('S')))), Over(Under(Ty('NP'), Ty('S')), Ty('NP'))),
        Box('FC((S << (NP >> S)), ((NP >> S) << NP))', Ty(Over(Ty('S'), Under(Ty('NP'), Ty('S'))), Over(Under(Ty('NP'), Ty('S')), Ty('NP'))), Over(Ty('S'), Ty('NP'))),
        FA(Over(Ty('NP'), Over(Ty('S'), Ty('NP')))),
        FA(Over(Under(Ty('NP'), Ty('S')), Ty('NP'))),
        BA(Under(Ty('NP'), Ty('S')))]
    assert diagram.offsets == [0, 1, 2, 1, 2, 3, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    from discopy.rigid import Ob, Ty, Box, Cup
    biclosed2rigid(diagram).boxes == [
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
        Cup(Ty('NP'), Ty(Ob('NP', z=1)))]
    biclosed2rigid(diagram).offsets ==\
        [0, 1, 4, 1, 4, 7, 7, 10, 13, 18, 17, 10, 9, 8, 6, 5, 3, 0]
