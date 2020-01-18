import os
from pytest import raises
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from discopy.pregroup import *
from discopy.rigidcat import Cap, RigidFunctor


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
    with raises(TypeError):
        draw(0)
    with raises(ValueError):
        draw(Box('s', Ty(), Ty()))
    dir, file = 'docs/imgs/', 'alice-loves-bob.png'
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    draw(sentence, show=False, fontsize=18, fontsize_types=12,
         figsize=(5, 2), margins=(0, 0))
    plt.savefig(dir + '.' + file)
    assert compare_images(dir + file, dir + '.' + file, 0) is None
    os.remove(dir + '.' + file)
    plt.clf()


def test_Diagram_to_gif():
    dir, file = 'docs/imgs/', 'autonomisation.gif'
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Box("Alice", Ty(), n), Box("Bob", Ty(), n)
    loves = Box('loves', Ty(), n.r @ s @ n.l)
    love_box = Box('loves', n @ n, s)
    love_ansatz = Cap(n.r, n) @ Cap(n, n.l) >> Id(n.r) @ love_box @ Id(n.l)
    ob, ar = {s: s, n: n}, {Alice: Alice, Bob: Bob, loves: love_ansatz}
    F = RigidFunctor(ob, ar)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    F(sentence).to_gif(
        dir + file, timestep=1000, loop=False, draw_types=True)
