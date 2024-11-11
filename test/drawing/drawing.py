
from pytest import raises

from discopy import utils
from discopy.utils import AxiomError
from discopy.compact import *
from discopy.drawing import *

IMG_FOLDER, TIKZ_FOLDER, TOL = 'test/drawing/imgs/', 'test/drawing/tikz/', 10

def draw_and_compare(file, **params):
    tol = params.pop('tol', TOL)
    return utils.draw_and_compare(file, IMG_FOLDER, tol, **params)


def tikz_and_compare(file, **params):
    return utils.tikz_and_compare(file, TIKZ_FOLDER, **params)


@draw_and_compare('crack-eggs.png')
def test_draw_eggs():
    def merge(x):
        return Box('merge', x @ x, x)

    egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
    crack = Box('crack', egg, white @ yolk)
    return crack @ crack\
        >> Id(white) @ Swap(yolk, white) @ Id(yolk)\
        >> merge(white) @ merge(yolk)

@draw_and_compare('crack-two-eggs-at-once.png')
def test_crack_two_eggs_at_once():
    from discopy.monoidal import Layer
    from discopy.symmetric import Ty, Box, Diagram

    egg, white, yolk = Ty("egg"), Ty("white"), Ty("yolk")
    crack = Box("crack", egg, white @ yolk)
    merge = lambda X: Box("merge", X @ X, X)

    # DisCoPy allows string diagrams to be defined as Python functions

    @Diagram.from_callable(egg @ egg, white @ yolk)
    def crack_two_eggs(x, y):
        (a, b), (c, d) = crack(x), crack(y)
        return (merge(white)(a, c), merge(yolk)(b, d))

    # ... or in point-free style using parallel (@) and sequential (>>) composition

    assert crack_two_eggs == crack @ crack\
        >> white @ Diagram.swap(yolk, white) @ yolk\
        >> merge(white) @ merge(yolk)

    crack_two_eggs_at_once = crack_two_eggs.foliation()

    assert crack_two_eggs_at_once == Diagram(
        dom=egg @ egg, cod=white @ yolk, inside=(
            Layer(Ty(), crack, Ty(), crack, Ty()),
            Layer(white, Diagram.swap(yolk, white), yolk),
            Layer(Ty(), merge(white), Ty(), merge(yolk), Ty())))

    return crack_two_eggs_at_once


@draw_and_compare("bubble-straight-wire.png", wire_labels=False)
def test_draw_bubble_wires():
    return (Ty('x') @ Box('s', Ty(), Ty())).bubble()


@draw_and_compare(
    'spiral.png', wire_labels=False,
    draw_box_labels=False, aspect='equal')
def test_draw_spiral():
    return spiral(2)


@draw_and_compare('who-ansatz.png', aspect='equal')
def test_draw_who():
    n, s = Ty('n'), Ty('s')
    copy, update = Box('copy', n, n @ n), Box('update', n @ s, s)
    return Cap(n.r, n)\
        >> Id(n.r) @ copy\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ update @ Id(s.l @ n)


@draw_and_compare('alice-loves-bob.png')
def test_draw_pregroup_sentence():
    from discopy.grammar.pregroup import Ty, Cup, Word, Id
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    return sentence.foliation()


@draw_and_compare('categorial-grammar.png', aspect='equal')
def test_draw_sentence():
    from discopy.closed import Ty
    from discopy.grammar.categorial import Word, BA, FA

    s, n = map(Ty, 'sn')

    Alice = Word('Alice', n)
    loves = Word('loves', (n >> s) << n)
    Bob = Word('Bob', n)

    return Alice @ loves @ Bob >> n @ FA((n >> s) << n) >> BA(n >> s)


@draw_and_compare('bialgebra.png', aspect='equal')
def test_draw_bialgebra():
    from discopy.quantum.zx import Z, X, Id, SWAP
    bialgebra = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    return bialgebra + bialgebra


@draw_and_compare("snake-equation.png",
                  aspect='auto', figsize=(5, 2), wire_labels=False)
def test_snake_equation():
    from discopy.rigid import Ty, Id
    x = Ty('x')
    return Equation(Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose())


@draw_and_compare('typed-snake-equation.png', figsize=(4, 1))
def test_draw_typed_snake():
    from discopy.rigid import Ty, Id
    x = Ty('x')
    return Equation(Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose())


@tikz_and_compare("spiral.tikz", wire_labels=False, use_tikzstyles=True)
def test_spiral_to_tikz():
    return spiral(2)


@tikz_and_compare("copy.tikz", use_tikzstyles=True)
def test_copy_to_tikz():
    x, y = map(Ty, ("$x$", "$y$"))
    copy_x, copy_y = Box('COPY', x, x @ x), Box('COPY', y, y @ y)
    copy_x.draw_as_spider, copy_y.draw_as_spider = True, True
    copy_x.drawing_name, copy_y.drawing_name = "", ""
    copy_x.color, copy_y.color = "black", "black"
    return copy_x @ copy_y >> Id(x) @ Swap(x, y) @ Id(y)


@draw_and_compare('empty_diagram.png')
def test_empty_diagram():
    return Id()


@draw_and_compare('bell-state.png', aspect='equal')
def test_draw_bell_state():
    from discopy.quantum import qubit, H, sqrt, Bra, Ket, CX
    return sqrt(2) >> Ket(0, 0) >> H @ qubit >> CX >> Bra(0) @ qubit


@tikz_and_compare("snake-equation.tikz", textpad=(.2, .2), textpad_words=(0, .25))
def test_snake_equation_to_tikz():
    from discopy.rigid import Ty, Id
    x = Ty('x')
    return Equation(Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose())


@tikz_and_compare("who-ansatz.tikz")
def test_who_ansatz_to_tikz():
    from discopy.grammar.pregroup import Ty, Cap, Word, Id, Box
    s, n = Ty('s'), Ty('n')
    who = Word('who', n.r @ n @ s.l @ n)
    who_ansatz = Cap(n.r, n)\
        >> Id(n.r) @ Box('copy', n, n @ n)\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ Box('update', n @ s, n) @ Id(s.l @ n)
    return Equation(who, who_ansatz, symbol="$\\mapsto$")


@tikz_and_compare('bialgebra.tikz', use_tikzstyles=True)
def test_tikz_bialgebra_law():
    from discopy.quantum.zx import Z, X, Id, SWAP
    source = X(2, 1) >> Z(1, 2)
    target = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    return Equation(source, target)


@tikz_and_compare('bell-state.tikz', aspect='equal', use_tikzstyles=True)
def test_tikz_bell_state():
    from discopy.quantum import qubit, H, sqrt, Bra, Ket, CX
    H.draw_as_spider, H.color, H.drawing_name = True, "yellow", ""
    return sqrt(2) >> Ket(0, 0) >> H @ qubit >> CX >> Bra(0) @ qubit


@tikz_and_compare('crack-eggs.tikz')
def test_tikz_eggs():
    def merge(x):
        box = Box('merge', x @ x, x, draw_as_spider=True)
        return box

    egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
    crack = Box('crack', egg, white @ yolk)
    return crack @ crack\
        >> Id(white) @ Swap(yolk, white) @ Id(yolk)\
        >> merge(white) @ merge(yolk)


@draw_and_compare('long-controlled.png', wire_labels=False, tol=5)
def test_draw_long_controlled():
    from discopy.quantum import Controlled, CZ, CX
    return (Controlled(CX.l, distance=3) >> Controlled(
        Controlled(CZ.l, distance=2), distance=-1))


@tikz_and_compare('long-controlled.tikz', wire_labels=False)
def test_tikz_long_controlled():
    from discopy.quantum import Controlled, CZ, CX
    return (Controlled(CX.l, distance=3) >> Controlled(
        Controlled(CZ.l, distance=2), distance=-1))

def test_to_gif():
    from discopy.grammar.pregroup import (
         Ty, Cup, Cap, Box, Word, Functor)

    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)

    sentence = Alice @ loves @ Bob\
        >> Cup(n, n.r) @ s @ Cup(n.l, n)

    def wiring(word):
        if word.cod == n:  # word is a noun
            return word
        if word.cod == n.r @ s @ n.l:  # word is a transitive verb
            box = Box(word.name, n @ n, s)
            return Cap(n.r, n) @ Cap(n, n.l) >> n.r @ box @ n.l

    W = Functor(ob={s: s, n: n}, ar=wiring)

    rewrite_steps = W(sentence).normalize()
    params = dict(
        path=IMG_FOLDER + 'autonomisation.gif', timestep=1000, figsize=(4, 4))
    sentence.to_gif(*rewrite_steps, **params)
