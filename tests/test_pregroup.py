import os
from pytest import raises
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from discopy.pregroup import *


def test_Word():
    with raises(TypeError):
        Word(0, Ty('n'))
    with raises(TypeError):
        Word('Alice', 0)


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


def test_draw():
    dir, file = 'docs/', 'alice-loves-bob.png'
    plt.rcParams.update({'font.size': 18, 'figure.figsize': (5.8, 2)})
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    draw(sentence, show=False)
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(dir + '.' + file)
    assert compare_images(dir + file, dir + '.' + file, 0) is None
    os.remove(dir + '.' + file)
