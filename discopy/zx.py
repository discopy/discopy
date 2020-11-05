# -*- coding: utf-8 -*-

""" Implements ZX diagrams. """

from discopy import messages, monoidal, rigid
from discopy.rigid import PRO, Diagram, Box


class Diagram(rigid.Diagram):
    def __init__(self, dom, cod, boxes, offsets, layers=None):
        super().__init__(dom, cod, boxes, offsets, layers)

    def __repr__(self):
        return super().__repr__().replace('Diagram', 'zx.Diagram')

    @staticmethod
    def upgrade(diagram):
        return Diagram(diagram.dom, diagram.cod,
                       diagram.boxes, diagram.offsets, layers=diagram.layers)

    @staticmethod
    def id(dom):
        return Id(len(dom))

    @staticmethod
    def sum(terms, dom=None, cod=None):
        return Sum(terms, dom, cod)

    @staticmethod
    def swap(left, right):
        return monoidal.swap(
            left, right, ar_factory=Diagram, swap_factory=Swap)

    @staticmethod
    def permutation(perm, dom=None):
        dom = PRO(len(perm)) if dom is None else dom
        return monoidal.permutation(perm, dom, ar_factory=Diagram)

    @staticmethod
    def cups(left, right):
        return rigid.cups(
            left, right, ar_factory=Diagram, cup_factory=lambda *_: Z(2, 0))

    @staticmethod
    def caps(left, right):
        return rigid.caps(
            left, right, ar_factory=Diagram, cap_factory=lambda *_: Z(0, 2))

    def draw(self, **params):
        return super().draw(**dict(params, draw_types=False))


class Id(rigid.Id, Diagram):
    """ Identity ZX diagram. """
    def __init__(self, dom):
        super().__init__(PRO(dom))

    def __repr__(self):
        return "Id({})".format(len(self.dom))

    __str__ = __repr__


class Swap(rigid.Swap, Diagram):
    """ Swap in a ZX diagram. """
    def __init__(self, left, right):
        if not isinstance(left, PRO):
            raise TypeError(messages.type_err(PRO, left))
        if not isinstance(right, PRO):
            raise TypeError(messages.type_err(PRO, right))
        super().__init__(left, right)

    def __repr__(self):
        return "SWAP"

    __str__ = __repr__


class Sum(monoidal.Sum, Diagram):
    """ Sum of ZX diagrams. """
    @staticmethod
    def upgrade(old):
        return Sum(old.terms, old.dom, old.cod)


class Spider(Box, Diagram):
    """ Spider boxes. """
    def __init__(self, name, n_legs_in, n_legs_out, phase=0):
        dom, cod = PRO(n_legs_in), PRO(n_legs_out)
        name = "{}({}, {}{})".format(
            name, n_legs_in, n_legs_out, ", {}".format(phase) if phase else "")
        Box.__init__(self, name, dom, cod, data=phase)
        Diagram.__init__(self, dom, cod, [self], [0])
        self.draw_as_spider, self.drawing_name = True, phase or ""

    @property
    def phase(self):
        """ Phase of a spider. """
        return self.data

    def __repr__(self):
        return self._name

    __str__ = __repr__


class Z(Spider):
    """ Z spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__("Z", n_legs_in, n_legs_out, phase)
        self.color = "green"


class X(Spider):
    """ X spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__("X", n_legs_in, n_legs_out, phase)
        self.color = "red"


SWAP = Swap(PRO(1), PRO(1))
