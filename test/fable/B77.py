"""B77: Int.caps is the wrong way round, trace and curry are borrowed stubs, Int(Matrix[bool]).braid does int @ int (discopy/interaction.py:396, 453, 337).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy import compact
from discopy.interaction import Diagram, Int, Ty
from discopy.matrix import Matrix

x0, x1, y0, y1 = map(compact.Ty, ["x0", "x1", "y0", "y1"])
X, Y = Ty[compact.Ty](x0, x1), Ty[compact.Ty](y0, y1)
D = Int(compact.Diagram)


def test_b77_cups_on_a_type_with_both_halves_control_passes():
    cups = D.cups(-X, X)
    assert cups.dom == -X @ X and cups.cod == Ty[compact.Ty]()


def test_b77_caps_on_a_type_with_both_halves():
    caps = D.caps(X, -X)
    assert caps.dom == Ty[compact.Ty]() and caps.cod == X @ -X


def test_b77_snake_on_a_type_with_both_halves():
    snake = D.caps(X, -X) @ X >> X @ D.cups(-X, X)
    assert snake.dom == snake.cod == X


def test_b77_trace_on_an_interaction_diagram():
    g = D(compact.Box('g', x0 @ y0 @ y1 @ x1, x0 @ y0 @ y1 @ x1), X @ Y, X @ Y)
    assert g.trace().dom == g.trace().cod == X


def test_b77_curry_on_an_interaction_diagram():
    f = D(compact.Box('f', x0 @ y1, y0 @ x1), X, Y)
    assert f.curry().dom == Ty[compact.Ty]() and f.curry().cod == Y @ -X


def test_b77_bool_matrix_braid():
    M, T = Diagram[Matrix[bool]], Ty[int]
    braid = M.braid(T(1, 1), T(1, 1))
    assert braid.dom == braid.cod == T(2, 2)
