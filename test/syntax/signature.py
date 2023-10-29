# -*- coding: utf-8 -*-

import networkx as nx
from discopy import signature


def test_merengue():
    sig = nx.DiGraph()
    # crack
    sig.add_edge('egg', 'crack')
    sig.add_edge('crack', 'white')
    sig.add_edge('crack', 'yolk')
    # mix
    sig.add_edge('white', 'mix')
    sig.add_edge('yolk', 'mix')
    sig.add_edge('mix', 'egg')
    signature.Arrow(sig).draw()

    crack = signature.Box(sig, 'crack')
    mix = signature.Box(sig, 'mix')

    crack_tensor_mix = crack @ mix
    crack_then_mix = crack >> mix

    from discopy.drawing import Equation
    Equation(crack_tensor_mix, crack_then_mix, symbol=' and ').draw(space=2, figsize=(8, 2))
