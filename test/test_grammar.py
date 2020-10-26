import os
from pytest import raises
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from discopy.grammar import *
from discopy.rigid import Cap, Ty


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
            'children': [{'word': 'This',
            'lemma': 'XX',
            'pos': 'XX',
            'entity': 'XX',
            'chunk': 'XX',
            'cat': 'NP'},
            {'type': 'ba',
            'cat': 'S[dcl]\\NP',
            'children': [{'type': 'fa',
             'cat': 'S[dcl]\\NP',
             'children': [{'word': 'is',
               'lemma': 'XX',
               'pos': 'XX',
               'entity': 'XX',
               'chunk': 'XX',
               'cat': '(S[dcl]\\NP)/NP'},
              {'type': 'fa',
               'cat': 'NP',
               'children': [{'word': 'a',
                 'lemma': 'XX',
                 'pos': 'XX',
                 'entity': 'XX',
                 'chunk': 'XX',
                 'cat': 'NP[nb]/N'},
                {'word': 'test',
                 'lemma': 'XX',
                 'pos': 'XX',
                 'entity': 'XX',
                 'chunk': 'XX',
                 'cat': 'N'}]}]},
            {'type': 'conj',
             'cat': '(S[dcl]\\NP)\\(S[dcl]\\NP)',
             'children': [{'word': 'and',
               'lemma': 'XX',
               'pos': 'XX',
               'entity': 'XX',
               'chunk': 'XX',
               'cat': 'conj'},
              {'type': 'fa',
               'cat': 'S[dcl]\\NP',
               'children': [{'word': 'learn',
                 'lemma': 'XX',
                 'pos': 'XX',
                 'entity': 'XX',
                 'chunk': 'XX',
                 'cat': '(S[dcl]\\NP)/NP'},
                {'type': 'lex',
                 'cat': 'NP',
                 'children': [{'word': 'sentence.',
                   'lemma': 'XX',
                   'pos': 'XX',
                   'entity': 'XX',
                   'chunk': 'XX',
                   'cat': 'N'}]}]}]}]}]}
    diagram = tree2diagram(tree)
    from discopy.biclosed import Ty, Over, Under, Box, FA, BA, Functor
    assert diagram.boxes == [
        CCGWord('This', Ty('NP')),
        CCGWord('is', Over(Under(Ty('NP'), Ty('S')), Ty('NP'))),
        CCGWord('a', Over(Ty('NP'), Ty('N'))),
        CCGWord('test', Ty('N')),
        FA(Over(Ty('NP'), Ty('N')), Ty('N')),
        FA(Over(Under(Ty('NP'), Ty('S')), Ty('NP')), Ty('NP')),
        CCGWord('and', Ty('conj')),
        CCGWord('learn', Over(Under(Ty('NP'), Ty('S')), Ty('NP'))),
        CCGWord('sentence.', Ty('N')),
        Box('lex', Ty('N'), Ty('NP')),
        FA(Over(Under(Ty('NP'), Ty('S')), Ty('NP')), Ty('NP')),
        Box('conj', Ty('conj', Under(Ty('NP'), Ty('S'))),
            Under(Under(Ty('NP'), Ty('S')), Under(Ty('NP'), Ty('S')))),
        BA(Under(Ty('NP'), Ty('S')),
           Under(Under(Ty('NP'), Ty('S')), Under(Ty('NP'), Ty('S')))),
        BA(Ty('NP'), Under(Ty('NP'), Ty('S')))]
    assert diagram.offsets == [0, 1, 2, 3, 2, 1, 2, 3, 4, 4, 3, 2, 1, 0]
    F_ob = Functor(
        ob=lambda x: x, ar={}, ob_factory=rigid.Ty, ar_factory=rigid.Diagram)
    F = Functor(
        ob=lambda x: x,
        ar=lambda f: rigid.Box(f.name, F_ob(f.dom), F_ob(f.cod)),
        ob_factory=rigid.Ty, ar_factory=rigid.Diagram)
    from discopy.rigid import Ob, Ty, Box, Cup
    F(diagram).boxes == [
        Box('This', Ty(), Ty('NP')),
        Box('is', Ty(), Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1))),
        Box('a', Ty(), Ty('NP', Ob('N', z=-1))),
        Box('test', Ty(), Ty('N')),
        Cup(Ty(Ob('N', z=-1)), Ty('N')),
        Cup(Ty(Ob('NP', z=-1)), Ty('NP')),
        Box('and', Ty(), Ty('conj')),
        Box('learn', Ty(), Ty(Ob('NP', z=1), 'S', Ob('NP', z=-1))),
        Box('sentence.', Ty(), Ty('N')),
        Box('lex', Ty('N'), Ty('NP')),
        Cup(Ty(Ob('NP', z=-1)), Ty('NP')),
        Box('conj', Ty('conj', Ob('NP', z=1), 'S'),
            Ty(Ob('S', z=1), Ob('NP', z=2), Ob('NP', z=1), 'S')),
        Cup(Ty('S'), Ty(Ob('S', z=1))),
        Cup(Ty(Ob('NP', z=1)), Ty(Ob('NP', z=2))),
        Cup(Ty('NP'), Ty(Ob('NP', z=1)))]
    F(diagram).offsets == [0, 1, 4, 6, 5, 3, 3, 4, 7, 7, 6, 3, 2, 1, 0]
