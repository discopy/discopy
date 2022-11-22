# -*- coding: utf-8 -*-

"""
The free pivotal category, i.e. rigid where left and right adjoints coincide.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Ty
    Layer
    Diagram
    Box
    Cup
    Cap
    Category
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        cups
        caps

Axioms
------

>>> unit, s, n = Ty(), Ty('s'), Ty('n')
>>> t = n.r @ s @ n.l
>>> assert t @ unit == t == unit @ t
>>> assert t.l.r == t == t.r.l
>>> left_snake, right_snake = Id(n.r).transpose(left=True), Id(n.l).transpose()
>>> assert left_snake.normal_form() == Id(n) == right_snake.normal_form()
>>> from discopy import drawing
>>> drawing.equation(
...     left_snake, Id(n), right_snake, figsize=(4, 2),
...     path='docs/_static/imgs/rigid/snake-equation.png')

.. image:: ../_static/imgs/rigid/snake-equation.png
    :align: center
"""

from discopy import rigid
from discopy.cat import factory
from discopy.rigid import nesting


class Ob(rigid.Ob):
    l = r = property(lambda self: self.cast(Ob(self.name, (self.z + 1) % 2)))

@factory
class Ty(rigid.Ty, Ob):
    def __init__(self, inside=()):
        rigid.Ty.__init__(self, inside=tuple(map(Ob.cast, inside)))

class Diagram(rigid.Diagram): pass

class Box(rigid.Box, Diagram):
    cast = Diagram.cast

class Cup(rigid.Cup, Box):
    def dagger(self):
        return Cap(self.dom[0], self.dom[1])

class Cap(rigid.Cap, Box):
    def dagger(self):
        return Cup(self.cod[0], self.cod[1])

Diagram.cups, Diagram.caps = nesting(Cup), nesting(Cap)

class Functor(rigid.Functor):
    dom = cod = Category(Ty, Diagram)
