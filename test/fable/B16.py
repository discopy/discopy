# -*- coding: utf-8 -*-
"""B16: the functor drops spider phases and Spider repr emits a phase= kwarg its __init__ rejects (discopy/frobenius.py:299, discopy/frobenius.py:261).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy import frobenius
from discopy.frobenius import Functor, Spider, Ty


def test_b16_functor_keeps_phase():
    sp = Spider(1, 1, Ty('x'), 0.5)
    identity = Functor(ob_map=lambda x: x, ar_map=lambda f: f)
    assert identity(sp) == sp


def test_b16_repr_roundtrip():
    sp = Spider(1, 1, Ty('x'), 0.5)
    assert eval(repr(sp), {'frobenius': frobenius}) == sp
