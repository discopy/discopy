import os
from pytest import raises

from PIL import Image, ImageChops
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from discopy import *


IMG_FOLDER, TIKZ_FOLDER, TOL = 'test/imgs/', 'test/tikz/', 10


def draw_and_compare(file, folder=IMG_FOLDER, tol=TOL,
                     draw=Diagram.draw, **params):
    def decorator(func):
        def wrapper():
            true_path = os.path.join(folder, file)
            test_path = os.path.join(folder, '.' + file)
            draw(func(), path=test_path, **params)
            test = compare_images(true_path, test_path, tol)
            assert test is None
            os.remove(test_path)
        return wrapper
    return decorator


@draw_and_compare('crack-eggs.png', figsize=(5, 6), fontsize=18)
def test_draw_eggs():
    def merge(x):
        return Box('merge', x @ x, x)
    egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
    crack = Box('crack', egg, white @ yolk)
    crack_two_eggs = crack @ crack\
        >> Id(white) @ Box('SWAP', yolk @ white, white @ yolk) @ Id(yolk)\
        >> merge(white) @ merge(yolk)
    return crack_two_eggs


@draw_and_compare('spiral.png', draw_types=False, draw_box_labels=False)
def test_draw_spiral():
    return monoidal.spiral(2)


@draw_and_compare('who-ansatz.png')
def test_draw_who():
    n, s = Ty('n'), Ty('s')
    copy, update = Box('copy', n, n @ n), Box('update', n @ s, s)
    return Cap(n.r, n)\
        >> Id(n.r) @ copy\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ update @ Id(s.l @ n)


@draw_and_compare('sentence-as-diagram.png')
def test_draw_sentence():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    return Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)


@draw_and_compare('alice-loves-bob.png', draw=grammar.draw,
                  fontsize=18, fontsize_types=12,
                  figsize=(5, 2), margins=(0, 0))
def test_pregroup_draw():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    return Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)


@draw_and_compare('bell-state.png', draw=Circuit.draw, draw_as_nodes=[0])
def test_draw_bell_state():
    return quantum.H @ quantum.Id(1) >> quantum.CX


def draw_equation(diagrams, **params):
    return drawing.equation(*diagrams, **params)


@draw_and_compare("snake-equation.png", draw=draw_equation,
                  aspect='auto', figsize=(5, 2), draw_types=False)
def test_snake_equation():
    x = Ty('x')
    return Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose()


@draw_and_compare('typed-snake-equation.png', draw=draw_equation,
                  figsize=(5, 2), aspect='auto')
def test_draw_typed_snake():
    x = Ty('x')
    return Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose()


def tikz_and_compare(file, folder=TIKZ_FOLDER, draw=Diagram.draw, **params):
    def decorator(func):
        def wrapper():
            true_path = os.path.join(folder, file)
            test_path = os.path.join(folder, '.' + file)
            draw(func(), path=test_path, **params)
            with open(true_path, "r") as true:
                with open(test_path, "r") as test:
                    assert true.read() == test.read()
            os.remove(test_path)
        return wrapper
    return decorator


@tikz_and_compare("spiral.tex", to_tikz=True)
def test_spiral_to_tikz():
    return monoidal.spiral(2)


@tikz_and_compare("copy.tex", to_tikz=True,
                  draw_as_nodes=True, draw_box_labels=False, color='black')
def test_copy_to_tikz():
    x, y, z = map(Ty, ("$x$", "$y$", "$z$"))
    return Box('COPY', x, x @ x) @ Box('COPY', y, y @ y)\
        >> Id(x) @ Box("SWAP", x @ y, y @ x) @ Id(y)


@tikz_and_compare("alice-loves-bob.tex", to_tikz=True, draw=grammar.draw,
                  textpad=(.2, .2), textpad_words=(0, .25))
def test_sentence_to_tikz():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    return Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)


@tikz_and_compare("snake-equation.tex", to_tikz=True, draw=draw_equation,
                  textpad=(.2, .2), textpad_words=(0, .25))
def test_snake_equation_to_tikz():
    x = Ty('x')
    return Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose()


@tikz_and_compare("who-ansatz.tex", to_tikz="controls",
                  draw=draw_equation, symbol="$\\mapsto$")
def test_who_ansatz_to_tikz():
    s, n = Ty('s'), Ty('n')
    who = Word('who', n.r @ n @ s.l @ n)
    who_ansatz = Cap(n.r, n)\
        >> Id(n.r) @ Box('copy', n, n @ n)\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ Box('update', n @ s, n) @ Id(s.l @ n)
    return who, who_ansatz
