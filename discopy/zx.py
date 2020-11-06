# -*- coding: utf-8 -*-

""" Implements ZX diagrams. """

from discopy import monoidal, rigid, Functor
from discopy.rigid import PRO, Diagram, Box

from discopy.quantum import Bra, Ket, Rz, Rx, H, CX, CZ, CRz, CRx
from discopy.quantum import Z as PauliZ
from discopy.quantum import X as PauliX
from discopy.quantum import Y as PauliY


class Diagram(rigid.Diagram):
    @staticmethod
    def upgrade(diagram):
        return Diagram(diagram.dom, diagram.cod,
                       diagram.boxes, diagram.offsets, layers=diagram.layers)

    def __init__(self, dom, cod, boxes, offsets, layers=None):
        super().__init__(dom, cod, boxes, offsets, layers)

    def draw(self, **params):
        return super().draw(**dict(params, draw_types=False))

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
        Diagram.__init__(self, dom, cod, [self], [0])
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
        self.color = "green"

class Y(Spider):
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        name = "X({}, {}, {})".format(n_legs_in, n_legs_out, phase)
        super().__init__(name, n_legs_in, n_legs_out, phase)
        self.color = "blue"

class X(Spider):
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        name = "X({}, {}, {})".format(n_legs_in, n_legs_out, phase)
        super().__init__(name, n_legs_in, n_legs_out, phase)
        self.color = "red"

class Had(Box, Diagram):
    def __init__(self):
        name = "Had()"
        dom, cod = PRO(1), PRO(1)
        Box.__init__(self, name, dom, cod)
        Diagram.__init__(self, dom, cod, [self], [0])
        self.draw_as_spider = True
        self.color = "yellow"

    @property
    def name(self):
        return self.data or "H"

    def __str__(self):
        return self._name

    def __repr__(self):
        return str(self)


SWAP = Swap(PRO(1), PRO(1))
BIALGEBRA = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)


def box2zx(box):
    from functools import reduce
    if isinstance(box, Bra):
        bits = box.bitstring
        return reduce(lambda a, b: a @ b, map(lambda p: X(1, 0, phase=p), bits))
    elif isinstance(box, Ket):
        bits = box.bitstring
        return reduce(lambda a, b: a @ b, map(lambda p: X(0, 1, phase=p), bits))
    elif isinstance(box, Rz):
        return Z(1,1, box.phase)
    elif isinstance(box, Rx):
        return X(1,1, box.phase)
    elif box == H:
        return Had()
    elif box == CX:
        return Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1)
    elif box == CZ:
        return Z(1, 2) @ Id(1) >> Id(1) @ Had() @ Id(1) >> Id(1) @ Z(2, 1)
    elif box == PauliZ:
        return Z(1,1,1)
    elif box == PauliY:
        return Y(1,1,1)
    elif box == PauliX:
        return X(1,1,1)
    elif isinstance(box, CRz):
        p = box.phase
        return Z(1, 2) @ Z(1, 2, p) >> Id(1) @ (X(2, 1) >> Z(1, 0, -p)) @ Id(1)
    elif isinstance(box, CRx):
        p = box.phase
        return X(1, 2) @ X(1, 2, p) >> Id(1) @ (Z(2, 1) >> X(1, 0, -p)) @ Id(1)
    elif isinstance(box, CU1):
        p = box.phase
        return Z(1, 2, p) @ Z(1, 2, p) >> Id(1) @ (X(2, 1) >> Z(1, 0, -p)) @ Id(1)
    

    return box

circuit2zx = Functor(
    ob=lambda x: x, ar=box2zx,
    ob_factory=rigid.Ty, ar_factory=rigid.Diagram)
