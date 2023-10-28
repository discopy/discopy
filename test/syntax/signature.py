# -*- coding: utf-8 -*-

import networkx as nx
from discopy import signature


def test_Arrow():
    sig = nx.DiGraph()
    sig.add_edge('x', 'x')
    sig.add_edge('w', 'f')
    sig.add_edge('x', 'f')
    sig.add_edge('f', 'y')
    sig.add_edge('y', 'g')
    sig.add_edge('g', 'z')

    ar = signature.Arrow(sig)
    f = signature.Box(sig, 'f')
    x = signature.Box(sig, 'x')
    (f @ ar @ x).draw()
