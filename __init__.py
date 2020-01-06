# -*- coding: utf-8 -*-

"""
discopy computes natural language meaning in pictures.

>>> s, n = Ty('s'), Ty('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
>>> ob, ar = {s: 1, n: 2}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = Model(ob, ar)
>>> assert F(Alice @ loves @ Bob >> grammar)
"""

from discopy import cat, moncat, pivotal, matrix, circuit, pregroup, config
from discopy.cat import Quiver, Functor
from discopy.moncat import MonoidalFunctor
from discopy.pivotal import Ob, Ty, Box, Diagram, Id, Cup, Cap, PivotalFunctor
from discopy.matrix import Dim, Matrix, MatrixFunctor
from discopy.circuit import PRO, Circuit, Gate, Bra, Ket, CircuitFunctor
from discopy.pregroup import Word, Model, CircuitModel

__version__ = config.VERSION
