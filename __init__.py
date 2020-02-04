# -*- coding: utf-8 -*-

"""
discopy computes natural language meaning in pictures.

>>> x = Ty('x')
>>> left_snake = Id(x) @ Cap(x.r, x) >> Cup(x, x.r) @ Id(x)
>>> right_snake =  Cap(x, x.l) @ Id(x) >> Id(x) @ Cup(x.l, x)
>>> assert left_snake.normal_form() == Id(x) == right_snake.normal_form()

>>> s, n = Ty('s'), Ty('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
>>> ob, ar = {s: 1, n: 2}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = MatrixFunctor(ob, ar)
>>> assert F(sentence)

>>> love_box = Box('loves', n @ n, s)
>>> love_ansatz = Cap(n.r, n) @ Cap(n, n.l) >> Id(n.r) @ love_box @ Id(n.l)
>>> ob, ar = {s: s, n: n}, {Alice: Alice, Bob: Bob, loves: love_ansatz}
>>> A = Functor(ob, ar)
>>> assert A(sentence).normal_form() == Alice @ Bob >> love_box
"""

from discopy import cat, moncat, rigidcat, matrix, circuit, pregroup
from discopy.cat import Quiver
from discopy.rigidcat import Ob, Ty, PRO, Box, Diagram, Id, Cup, Cap, Functor
from discopy.matrix import Dim, Matrix, MatrixFunctor
from discopy.circuit import Circuit, Gate, Bra, Ket, CircuitFunctor
from discopy.pregroup import Word

__version__ = '0.2.3'
