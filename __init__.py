# -*- coding: utf-8 -*-

"""
discopy computes natural language meaning in pictures.

>>> s, n = Ty('s'), Ty('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> print(Alice @ loves @ Bob)
Alice >> Wire(n) @ loves >> Wire(n @ n.r @ s @ n.l) @ Bob
"""

from discopy import cat, moncat, matrix, circuit, gates, disco, config
from discopy.cat import Quiver
from discopy.matrix import Dim, Matrix, MatrixFunctor
from discopy.circuit import PRO, Circuit, CircuitFunctor
from discopy.gates import Gate, Bra, Ket
from discopy.pregroup import Ob, Ty, Box, Diagram, Wire, Cup, Cap
from discopy.disco import Word, Model, CircuitModel

__version__ = config.VERSION
