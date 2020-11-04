# -*- coding: utf-8 -*-

""" Implements ZX diagrams. """

from discopy import monoidal, rigid
from discopy.rigid import PRO, Diagram, Box


class Diagram(rigid.Diagram):
    @staticmethod
    def upgrade(diagram):
        return ZX(diagram.dom, diagram.cod,
                  diagram.boxes, diagram.offsets, layers=diagram.layers)

    def __init__(self, dom, cod, boxes, offsets, layers=None):
        super().__init__(dom, cod, boxes, offsets, layers)

    def draw(self, **params):
        super().draw(**dict(params, draw_types=False))

    @staticmethod
    def id(dom):
        return Id(len(dom))


class Id(rigid.Id, Diagram):
    """ Identity ZX diagram. """
    def __init__(self, dom):
        super().__init__(PRO(dom))


class Swap(rigid.Swap, Diagram):
    """ Swap in a ZX diagram. """


class Sum(monoidal.Sum, Diagram):
    """ Sum of ZX diagrams. """


class Spider(Box, Diagram):
    def __init__(self, name, n_legs_in, n_legs_out, phase=0):
        dom, cod = PRO(n_legs_in), PRO(n_legs_out)
        Box.__init__(self, name, dom, cod, data=phase)
        ZX.__init__(self, dom, cod, [self], [0])
        self.draw_as_spider = True

    @property
    def phase(self):
        return self.data

    @property
    def name(self):
        return self.data or ""

    def __str__(self):
        return self._name

    def __repr__(self):
        return str(self)


class Z(Spider):
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        name = "Z({}, {}, {})".format(n_legs_in, n_legs_out, phase)
        super().__init__(name, n_legs_in, n_legs_out, phase)
        self.color = "red"


class X(Spider):
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        name = "X({}, {}, {})".format(n_legs_in, n_legs_out, phase)
        super().__init__(name, n_legs_in, n_legs_out, phase)
        self.color = "green"


SWAP = Swap(PRO(1), PRO(1))
BIALGEBRA = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
