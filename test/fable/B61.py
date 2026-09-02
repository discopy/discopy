"""B61: to_staircases, foliation and Functor.id rebuild bubbles through cod.bubble, dropping name and drawing flags (discopy/cat.py:927).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy.monoidal import Ty, Box, Functor

x, y = Ty('x'), Ty('y')
f, h = Box('f', x, y), Box('h', x, x)


def test_b61_foliation_keeps_bubble_name():
    bubble = f.bubble(name="N")
    assert (bubble @ h).foliation().boxes[0] == bubble


def test_b61_foliation_keeps_drawing_name():
    bubble = f.bubble(drawing_name="LABEL")
    assert (bubble @ h).foliation().boxes[0].drawing_name == "LABEL"


def test_b61_to_staircases_is_identity_on_named_bubble():
    bubble = f.bubble(name="N")
    assert bubble.to_staircases() == bubble


def test_b61_identity_functor_keeps_bubble_name():
    bubble = f.bubble(name="N")
    assert Functor.id()(bubble) == bubble


def test_b61_identity_functor_keeps_draw_as_frame():
    assert Functor.id()(f.bubble(draw_as_frame=True)).draw_as_frame
