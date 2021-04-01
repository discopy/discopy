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

from networkx import Graph, connected_components

from discopy import cat


def relabel(wires):
    ordered = sorted(set(wires), key=lambda i: wires.index(i))
    return [ordered.index(i) for i in wires]


class Diagram(cat.Arrow):
    """ Diagram in a hypergraph category. """
    def __init__(self, dom, cod, boxes, wires, n_spiders=None):
        super().__init__(dom, cod, boxes, _scan=False)
        if len(wires) != len(dom)\
                + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod):
            raise ValueError
        nodes = set(wires) if n_spiders is None else set(range(n_spiders))
        if not set(wires) <= nodes:
            raise ValueError
        if nodes != set(range(len(nodes))):
            raise ValueError
        if relabel(wires) != wires:
            raise ValueError
        self.nodes, self.wires = nodes, wires

    def then(self, other):
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod, boxes = self.dom, other.cod, self.boxes + other.boxes
        self_boundary = self.wires[len(self.wires) - len(self.cod):]
        other_boundary = other.wires[:len(other.dom)]
        graph = Graph([
            (("b", i), ("self", j)) for i, j in enumerate(self_boundary)
            ] + [
            (("b", i), ("other", j)) for i, j in enumerate(other_boundary)])
        components, self_pushout, other_pushout = set(), dict(), dict()
        for i, component in enumerate(connected_components(graph)):
            components.add(i)
            for case, j in component:
                if case == "self":
                    self_pushout[j] = i
                if case == "other":
                    other_pushout[j] = i
        self_nodes = self.nodes - set(self_boundary)
        self_pushout.update({
            j: len(components) + j
            for i, j in enumerate(self.wires) if j in self_nodes})
        other_nodes = other.nodes - set(other_boundary)
        other_pushout.update({
            j: len(components) + len(self_nodes) + j
            for i, j in enumerate(other.wires) if j in other_nodes})
        wires = [
            self_pushout[i] for i in self.wires[:
                len(self.wires) - len(self.cod)]]\
            + [other_pushout[i] for i in other.wires[len(other.dom):]]
        n_spiders = len(self.nodes) - len(set(self_boundary))\
            + len(components) + len(other.nodes) - len(set(other_boundary))
        return Diagram(dom, cod, boxes, relabel(wires), n_spiders)

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
        n_spiders = len(self.nodes) + len(other.nodes)
        return Diagram(dom, cod, boxes, wires, n_spiders)

    __matmul__ = tensor

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ['dom', 'cod', 'boxes', 'wires'])

    def __repr__(self):
        data = list(map(repr, [self.dom, self.cod, self.boxes, self.wires]))
        data += [", n_spiders={}".format(len(self.nodes))\
            if set(self.nodes) != set(self.wires) else ""]
        return "Diagram({}, {}, {}, {}{})".format(*data)


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
        boxes, wires = [], list(range(len(dom)))\
            + list(range(len(left), len(dom))) + list(range(len(left)))
        super().__init__(dom, cod, boxes, wires)


class Spider(Diagram):
    """ Spider diagram. """
    def __init__(self, n_legs_in, n_legs_out, typ):
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        boxes, n_spiders = [], len(typ)
        wires = (n_legs_in + n_legs_out) * list(range(n_spiders))
        super().__init__(dom, cod, boxes, wires, n_spiders)


Diagram.id = Id
Diagram.swap = Swap
Diagram.spiders = Spider
