import os
from pytest import raises
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from discopy.grammar import *
from discopy.rigid import Cap


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
    cfg = CFG(R0, R1, Jane, loves)
    assert Jane in cfg.productions
    assert "CFG(Box('R0', Ty('VP', 'N'), Ty('S'))" in repr(cfg)
    gen = cfg.generate(s, 2, 10, remove_duplicates=True, not_twice=[Jane, Bob])
    for sentence in gen:
        assert Jane in sentence.boxes
        assert Bob in sentence.boxes


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
    assert str(err.value) is messages.expected_pregroup()
