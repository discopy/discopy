import os
import pickle
from pytest import raises
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from discopy import rigid, messages
from discopy.utils import from_tree
from discopy.rigid import Id, Cup, Cap, Ty, Box
from discopy.biclosed import biclosed2rigid
from discopy.grammar import *


def test_Word():
    with raises(TypeError):
        Word(0, Ty('n'))
    with raises(TypeError):
        Word('Alice', 1)
    with raises(TypeError):
        Word('Alice', Ty('n'), dom=1)


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


def test_normal_form():
    n = Ty('n')
    w1, w2 = Word('a', n), Word('b', n)
    diagram = w1 @ w2 >>\
        Id(n) @ Cap(n, n.r) @ Id(n) >> Id(n @ n) @ Cup(n.r, n)
    expected_result = w1 @ w2
    assert expected_result == normal_form(diagram)

    with raises(ValueError) as err:
        normal_form(w2 >> w1 @ Id(n))


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
    tree, boxes, offsets, rigid_boxes, rigid_offsets =\
        pickle.load(open("test/src/tree2diagram.pickle", "rb"))
    diagram = tree2diagram(tree)
    rigid_diagram = biclosed2rigid(diagram)
    assert diagram.boxes == boxes
    assert diagram.offsets == offsets
    assert rigid_diagram.boxes == rigid_boxes
    assert rigid_diagram.offsets == rigid_offsets


def test_from_tree():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    sentence == from_tree(sentence.to_tree())
