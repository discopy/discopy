"""B76: no level between monoidal and frobenius sets bubble_factory, so a bubble composes with nothing there, and frobenius.Bubble has no z, so a bubbled diagram cannot rotate (discopy/monoidal.py:1759, frobenius.py:285).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy import (
    balanced, biclosed, braided, closed, compact, frobenius, markov, monoidal,
    pivotal, ribbon, rigid, symmetric, traced)

LEVELS = [
    monoidal, braided, balanced, symmetric, markov, traced, closed, biclosed,
    ribbon, rigid, pivotal, compact, frobenius]


@pytest.mark.parametrize(
    "module", LEVELS, ids=lambda module: module.__name__.split('.')[-1])
def test_b76_bubble_composes(module):
    """monoidal and frobenius are passing controls, the eleven others fail."""
    x, y = map(module.Ty, "xy")
    f, h = module.Box('f', x, y), module.Box('h', y, y)
    assert (f.bubble() >> h).cod == y


def test_b76_frobenius_bubble_rotates():
    a, b = map(frobenius.Ty, "ab")
    d = frobenius.Box('f', a, b).bubble() @ a
    assert (d.r.dom, d.r.cod) == (d.cod.r, d.dom.r)
