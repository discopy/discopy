
@draw_and_compare('bell-state.png', draw=Circuit.draw, aspect='equal')
def test_draw_bell_state():
    from discopy.quantum import qubit, H, sqrt, Bra, Ket, Id, CX
    return sqrt(2) >> Ket(0, 0) >> H @ qubit >> CX >> Bra(0) @ qubit


@tikz_and_compare("snake-equation.tikz", draw=draw_equation,
                  textpad=(.2, .2), textpad_words=(0, .25))
def test_snake_equation_to_tikz():
    x = Ty('x')
    return Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose()


@tikz_and_compare("who-ansatz.tikz",
                  draw=draw_equation, symbol="$\\mapsto$")
def test_who_ansatz_to_tikz():
    from discopy.grammar.pregroup import Ty, Cup, Cap, Word, Id, Box
    s, n = Ty('s'), Ty('n')
    who = Word('who', n.r @ n @ s.l @ n)
    who_ansatz = Cap(n.r, n)\
        >> Id(n.r) @ Box('copy', n, n @ n)\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ Box('update', n @ s, n) @ Id(s.l @ n)
    return who, who_ansatz


@tikz_and_compare('bialgebra.tikz', draw=draw_equation, use_tikzstyles=True)
def test_tikz_bialgebra_law():
    from discopy.quantum.zx import Z, X, Id, SWAP
    source = X(2, 1) >> Z(1, 2)
    target = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    return source, target


@tikz_and_compare('bell-state.tikz', aspect='equal', use_tikzstyles=True)
def test_tikz_bell_state():
    from discopy.quantum import qubit, H, sqrt, Bra, Ket, Id, CX
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


@draw_and_compare('long-controlled.png', draw_type_labels=False, tol=5)
def test_draw_long_controlled():
    from discopy.quantum import Controlled, CZ, CX
    return (
        Controlled(CX.l, distance=3)
        >> Controlled(Controlled(CZ.l, distance=2), distance=-1))


@tikz_and_compare('long-controlled.tikz', draw_type_labels=False)
def test_tikz_long_controlled():
    from discopy.quantum import Controlled, CZ, CX
    return (
        Controlled(CX.l, distance=3)
        >> Controlled(Controlled(CZ.l, distance=2), distance=-1))
