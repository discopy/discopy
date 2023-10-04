# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
from discopy import cat, monoidal, signature
from discopy.utils import AxiomError


def test_Arrow_str():
    sig = nx.DiGraph()
    sig.add_edge('w', 'f')
    sig.add_edge('x', 'f')
    sig.add_edge('f', 'y')
    sig.add_edge('y', 'g')
    sig.add_edge('g', 'z')

    f = signature.Box(sig, 'f')
    g = signature.Box(sig, 'g')
    (f >> g).draw()
