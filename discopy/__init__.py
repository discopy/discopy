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
>>> F = TensorFunctor(ob, ar)
>>> assert F(sentence)

>>> love_box = Box('loves', n @ n, s)
>>> love_ansatz = Cap(n.r, n) @ Cap(n, n.l) >> Id(n.r) @ love_box @ Id(n.l)
>>> ob, ar = {s: s, n: n}, {Alice: Alice, Bob: Bob, loves: love_ansatz}
>>> A = Functor(ob, ar)
>>> assert A(sentence).normal_form() == Alice @ Bob >> love_box
"""


from discopy import (
    config, cat, monoidal, rigid, biclosed, tensor, quantum, grammar)
from discopy.utils import from_tree
from discopy.cat import Quiver
from discopy.monoidal import Sum
from discopy.rigid import (
    Ob, Ty, PRO, Box, Diagram, Id, Cup, Cap, Swap, Functor)
from discopy.tensor import Dim, Tensor, Functor as TensorFunctor

from discopy.quantum import cqmap, zx
from discopy.quantum.zx import Z, X, H, SWAP
from discopy.quantum.cqmap import C, Q, CQMap
from discopy.quantum.circuit import (
    bit, qubit, Circuit, Functor as CircuitFunctor,
    Discard, MixedState, Measure, Encode)
from discopy.quantum.gates import (
    Ket, Bra, Bits, Copy, Match, Rx, Rz, CU1, CRz, CRx, CZ, CX)

from discopy.grammar import cfg, ccg, pregroup
from discopy.grammar.pregroup import Word


__version__ = '0.3.7.2'
