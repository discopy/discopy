"""B74: Hypergraph.caps demands the cup convention left.r == right, so every rigid cap is rejected and a pregroup diagram with a cap cannot be foliated (discopy/hypergraph.py:475).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy.grammar import pregroup
from discopy.hypergraph import Hypergraph

H = Hypergraph[pregroup.Diagram]
s, n = pregroup.Ty('s'), pregroup.Ty('n')


def test_b74_right_cap():
    h = H.caps(n.r, n)
    assert (h.dom, h.cod, h.boxes) == (pregroup.Ty(), n.r @ n, ())


def test_b74_left_cap():
    h = H.caps(n, n.l)
    assert (h.dom, h.cod, h.boxes) == (pregroup.Ty(), n @ n.l, ())


def test_b74_cap_to_hypergraph():
    h = pregroup.Cap(n.r, n).to_hypergraph()
    assert (h.dom, h.cod, h.boxes) == (pregroup.Ty(), n.r @ n, ())


def test_b74_cup_to_hypergraph_control():
    h = pregroup.Cup(n.l, n).to_hypergraph()
    assert (h.dom, h.cod, h.boxes) == (n.l @ n, pregroup.Ty(), ())


def test_b74_readme_wiring_foliates():
    loves = pregroup.Box('loves', n @ n, s)
    d = pregroup.Cap(n.r, n) @ pregroup.Cap(n, n.l) >> n.r @ loves @ n.l
    assert d.foliation().boxes == d.boxes
