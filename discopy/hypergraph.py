# -*- coding: utf-8 -*-

"""
Hypergraph categories.

Note
----

We can check spider fusion, i.e. special commutative Frobenius algebra.

>>> from discopy.monoidal import Ty
>>> x = Ty('x')
>>> split, merge = Spider(1, 2, x), Spider(2, 1, x)
>>> unit, counit = Spider(0, 1, x), Spider(1, 0, x)

Monoid and comonoid:

>>> assert unit @ Id(x) >> merge == Id(x) == Id(x) @ unit >> merge
>>> assert merge @ Id(x) >> merge == Id(x) @ merge >> merge
>>> assert split >> counit @ Id(x) == Id(x) == split >> Id(x) @ counit
>>> assert split >> split @ Id(x) == split >> Id(x) @ split

Frobenius:

>>> assert split @ Id(x) >> Id(x) @ merge\\
...     == merge >> split\\
...     == Id(x) @ split >> merge @ Id(x)\\
...     == Spider(2, 2, x)

Speciality:

>>> assert split >> merge == Spider(1, 1, x) == Id(x)

Coherence:

>>> assert Spider(0, 1, x @ x) == unit @ unit
>>> assert Spider(2, 1, x @ x) == Id(x) @ Swap(x, x) @ Id(x) >> merge @ merge
>>> assert Spider(1, 0, x @ x) == counit @ counit
>>> assert Spider(1, 2, x @ x) == split @ split >> Id(x) @ Swap(x, x) @ Id(x)
"""


from discopy import cat


def relabel(wires):
    ordered = sorted(set(wires), key=lambda i: wires.index(i))
    return [ordered.index(i) for i in wires]


class Diagram(cat.Arrow):
    """ Diagram in a hypergraph category. """
    def __init__(self, dom, cod, boxes, wires):
        super().__init__(dom, cod, boxes, _scan=False)
        if len(wires) != len(dom)\
                + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod):
            raise ValueError
        nodes = set(wires)
        if nodes != set(range(len(nodes))):
            raise ValueError
        if relabel(wires) != wires:
            raise ValueError
        self.nodes, self.wires = nodes, wires

    def then(self, other):
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod, boxes = self.dom, other.cod, self.boxes + other.boxes
        self_boundary = set(
            self.wires[-len(self.cod) + i] for i, _ in enumerate(self.cod))
        other_boundary = set(other.wires[i] for i, _ in enumerate(other.dom))
        self_nodes = self.nodes - self_boundary
        other_nodes = other.nodes - other_boundary
        nodes = list(range(len(self_nodes) + len(self.cod) + len(other_nodes)))
        self_pushout = {j: i for i, j in enumerate(self_nodes)}
        self_pushout.update({
            self.wires[-len(self.cod) + i]: len(self_nodes) + i
            for i, _ in enumerate(self.cod)})
        other_pushout = {
            other.wires[i]: len(self_nodes) + i
            for i, _ in enumerate(other.dom)}
        other_pushout.update({
            j: len(self_nodes) + len(self.cod) + i
            for i, j in enumerate(other_nodes)})
        wires = [
            self_pushout[i] for i in self.wires[:
                len(self.wires) - len(self.cod)]]\
            + [other_pushout[i] for i in other.wires[len(other.dom):]]
        return Diagram(dom, cod, boxes, relabel(wires))

    def tensor(self, other):
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        dom_wires = self.wires[:len(self.dom)] + [
            len(self.nodes) + i for i in other.wires[:len(other.dom)]]
        box_wires = self.wires[len(self.dom):-len(self.cod) or len(self.wires)]
        box_wires += [len(self.nodes) + i for i in other.wires[
            len(other.dom):-len(other.cod) or len(other.wires)]]
        cod_wires = self.wires[len(self.wires) - len(self.cod):] + [
            len(self.nodes) + i
            for i in other.wires[len(other.wires) - len(other.cod):]]
        wires = relabel(dom_wires + box_wires + cod_wires)
        return Diagram(dom, cod, boxes, wires)

    __matmul__ = tensor

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ['dom', 'cod', 'boxes', 'wires'])

    def __repr__(self):
        return "Diagram({}, {}, {}, {})".format(*map(repr, [
            self.dom, self.cod, self.boxes, self.wires]))


class Box(cat.Box, Diagram):
    """ Box in a :class:`discopy.hypergraph.Diagram`. """
    def __init__(self, name, dom, cod, **params):
        cat.Box.__init__(self, name, dom, cod, **params)
        boxes, nodes = [self], list(range(len(dom @ cod)))
        wires = 2 * list(range(len(dom)))\
            + 2 * list(range(len(dom), len(dom @ cod)))
        Diagram.__init__(self, dom, cod, boxes, wires)


class Id(Diagram):
    """ Identity diagram. """
    def __init__(self, dom):
        super().__init__(dom, dom, [], 2 * list(range(len(dom))))


class Swap(Diagram):
    """ Swap diagram. """
    def __init__(self, left, right):
        dom, cod = left @ right, right @ left
        boxes, nodes = [], list(range(len(dom)))
        wires = nodes\
            + list(range(len(left), len(dom))) + list(range(len(left)))
        super().__init__(dom, cod, boxes, wires)


class Spider(Diagram):
    """ Spider diagram. """
    def __init__(self, n_legs_in, n_legs_out, typ):
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        boxes, wires = [], len(dom @ cod) * [0]
        super().__init__(dom, cod, boxes, wires)


Diagram.id = Id
Diagram.swap = Swap
Diagram.spiders = Spider
