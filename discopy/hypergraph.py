# -*- coding: utf-8 -*-

""" Hypergraph categories.

Notes
-----

We can check that the axioms for symmetry hold on the nose.

>>> x, y, z = types("x y z")

Involution:

>>> assert Swap(x, y) >> Swap(y, x) == Id(x @ y)

Pentagons:

>>> assert Swap(x, y @ z) == Swap(x, y) @ Id(z) >> Id(y) @ Swap(x, z)
>>> assert Swap(x @ y, z) == Id(x) @ Swap(y, z) >> Swap(x, z) @ Id(y)

Yang-Baxter:

>>> left = Swap(x, y) @ Id(z)\\
...     >> Id(y) @ Swap(x, z)\\
...     >> Swap(y, z) @ Id(x)
>>> right = Id(x) @ Swap(y, z)\\
...     >> Swap(x, z) @ Id(y)\\
...     >> Id(z) @ Swap(x, y)
>>> assert left == right

Naturality:

>>> f = Box("f", x, y)
>>> assert f @ Id(z) >> Swap(f.cod, z) == Swap(f.dom, z) >> Id(z) @ f

We can also check spider fusion, i.e. special commutative Frobenius algebra.

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

import networkx as nx
from networkx import Graph, subgraph_view as subgraph

from discopy import cat, monoidal
from discopy.cat import AxiomError
from discopy.monoidal import Ty, types
from discopy.drawing import Node


Graph.relabel = nx.relabel_nodes


class Diagram(cat.Arrow):
    """
    Diagram in a hypergraph monoidal category.

    >>> x, y, z = types("x y z")
    >>> f, g = Box("f", x, y @ z), Box("g", z @ y, x)
    >>> diagram = f >> Swap(y, z) >> g
    >>> assert set(diagram.nodes) == set(range(
    ...     len(diagram.dom)
    ...     + sum(len(box.dom @ box.cod) for box in diagram.boxes)
    ...     + len(diagram.cod)))
    >>> diagram.edges
    EdgeView([(0, 1), (2, 5), (3, 4), (6, 7)])
    """
    def __init__(self, dom, cod, boxes, graph, _scan=True):
        super().__init__(dom, cod, boxes, _scan=False)
        if _scan:
            n_nodes = len(dom) + sum(len(box.dom) for box in boxes)
            n_nodes += sum(len(box.cod) for box in boxes) + len(cod)
            assert set(graph.nodes) == set(range(n_nodes))
            for i, j in graph.edges:
                assert i < j

        self._graph = graph

    @property
    def graph(self):
        """ The graph defining a diagram. """
        return self._graph

    @property
    def nodes(self):
        """ The nodes in the graph defining a diagram. """
        return self.graph.nodes

    @property
    def edges(self):
        """ The edges in the graph defining a diagram. """
        return self.graph.edges

    def __repr__(self):
        return "Diagram({}, {}, {}, Graph({}))".format(*map(repr, [
            self.dom, self.cod, self.boxes, list(self.edges)]))

    def then(self, other):
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        graph = nx.compose(self.graph, other.graph.relabel({
            i: len(self.nodes) - len(self.cod) + i for i in other.nodes}))
        boundary = range(len(self.nodes) - len(self.cod), len(self.nodes))
        for i in boundary:
            for source in graph.neighbors(i):
                for target in graph.neighbors(i):
                    if source < target:
                        graph.add_edge(source, target)
            graph.remove_node(i)
        graph = graph.relabel({
            i: i - len(boundary) for i in graph.nodes if i > boundary[-1]})
        return Diagram(dom, cod, boxes, graph)

    def tensor(self, other):
        """ Tensor product of two hypergraph monoidal diagrams. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        # move self.cod to the end
        self_graph = self.graph.relabel({
            i: len(other.nodes) - len(other.cod) + i
            for i in range(len(self.nodes) - len(self.cod), len(self.nodes))})
        # make space for other.dom
        self_graph = self_graph.relabel({i: i + len(other.dom) for i in range(
            len(self.dom), len(self.nodes) - len(self.cod))})
        # move other.dom to the start, other.cod to the end
        # and the nodes for other.boxes to just before self.cod
        other_graph = other.graph.relabel({
            i: len(self.dom) + i if i < len(other.dom)
            else len(self.nodes) + i if i >= len(other.nodes) - len(other.cod)
            else len(self.nodes) - len(self.cod) + i
            for i in range(len(other.nodes))})
        graph = nx.union(self_graph, other_graph)
        return Diagram(dom, cod, boxes, graph)

    def __matmul__(self, other):
        return self.tensor(other)

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return self.dom == other.dom\
            and self.cod == other.cod\
            and self.boxes == other.boxes\
            and set(self.graph.edges) == set(other.graph.edges)


class Box(cat.Box, Diagram):
    """ Box in a hypergraph monoidal diagram.

    Examples
    --------
    >>> x, y, z = types("x y z")
    >>> f = Box("f", x, y @ z)
    >>> f
    Box('f', Ty('x'), Ty('y', 'z'))
    """
    def __init__(self, name, dom, cod, **params):
        cat.Box.__init__(self, name, dom, cod, **params)
        boxes, graph = [self], Graph(
            [(i, len(dom) + i) for i, _ in enumerate(dom)] + [
                (len(dom @ dom) + i, len(dom @ dom @ cod) + i)
                for i, _ in enumerate(cod)])
        Diagram.__init__(self, dom, cod, [self], graph)


class Swap(Diagram):
    """ Swap in a hypergraph monoidal diagram. """
    def __init__(self, left, right):
        dom, cod = left @ right, right @ left
        boxes, graph = [], Graph(
            [(i, len(dom @ right) + i) for i, _ in enumerate(left)] + [
                (len(left) + i, len(dom) + i) for i, _ in enumerate(right)])
        super().__init__(dom, cod, boxes, graph)


class Id(Diagram):
    """ Identity diagram. """
    def __init__(self, dom):
        dom, cod, boxes, graph = dom, dom, [], Graph(
            [(i, len(dom) + i) for i, _ in enumerate(dom)])
        super().__init__(dom, cod, boxes, graph)


class Spider(Diagram):
    """ Spider diagrams, i.e. special commutative Frobenius algebra. """
    def __init__(self, n_legs_in, n_legs_out, typ):
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        boxes, graph = [], Graph()
        graph.add_nodes_from(range(len(dom @ cod)))
        for i, _ in enumerate(typ):
            nodes = [len(typ) * j + i for j in range(n_legs_in)]
            nodes += [len(dom) + len(typ) * j + i for j in range(n_legs_out)]
            graph.add_edges_from(
                (i, j) for i in nodes for j in nodes if i < j)
        super().__init__(dom, cod, boxes, graph)
