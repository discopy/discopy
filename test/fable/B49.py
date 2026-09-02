"""B49: Layer.merge normalises with the polymorphic normal_form, so foliation yanks snakes and dies on the README's own snake (discopy/monoidal.py:836).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy.rigid import Box, Cap, Cup, Ty

x, y = Ty('x'), Ty('y')
g = Box('g', y, y)
left_snake = x @ Cap(x.r, x) >> Cup(x, x.r) @ x


def test_b49_foliation_keeps_the_snake():
    d = g @ left_snake
    assert d.foliation().boxes == d.boxes


def test_b49_readme_snake_foliates():
    assert left_snake.foliation().boxes == left_snake.boxes
